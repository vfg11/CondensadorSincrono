import sys as _sys, os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ['01_model', '02_linearization', '03_design', '04_simulation']:
    _p = _os.path.join(_PKG_ROOT, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
OUTPUTS_DIR = _os.path.join(_PKG_ROOT, 'outputs')
_os.makedirs(OUTPUTS_DIR, exist_ok=True)
"""
reduce_and_design_pi.py
==========================
Two steps, both against the SAME single-output configuration confirmed
in chat: Q_delivered as measured (Qdeliv_m, the lagged state, NOT the
instantaneous algebraic Q_delivered -- Qdeliv_m is what a real sensor
chain would actually hand the controller).

1) ORDER REDUCTION: Hankel singular values on the 11-state linear
   model, output = Qdeliv_m only. Clear knee at order 6 (factor ~128x
   drop from sigma_6 to sigma_7, vs 2-6x between the others) -- the 5
   dropped states are the fast subtransient pair (poles near -90,
   Tdo''=0.03s/Tqo''=0.05s-linked) plus Vdeliv_m, which turns out to be
   EXACTLY unobservable from Q alone (sigma_11=0.0 exactly: Vdeliv_m
   drives nothing else and doesn't appear in Qdeliv_m's own dynamics,
   so watching Q can never reveal anything about it -- not a numerical
   coincidence, a structural fact about this state). matchdc balanced
   truncation (control.balred, needs slycot) to preserve DC gain
   exactly -- verified: open-loop step response of the order-6 model
   matches the full 11-state model to within 5.6e-5 over a 3s window.

2) PI DESIGN: grid search (Kp, Ki) on the order-6 reduced model's
   closed loop (standard PI-with-integrator augmentation), per chat:
   "agresiva/optimo establecimiento" -- minimise 2%-band settling time.
   A fine local grid found a well-behaved sweet spot (fast AND low
   overshoot together, not a speed-vs-overshoot trade needing a cap) --
   see chat for the coarse-grid intermediate result before refinement.

   This is the LINEAR/REDUCED-model tuning step only. It does NOT know
   about Vref's VREF_MAX=1.15 clamp, the AVR's own internal saturations,
   or the outer loop's own anti-windup -- validate_pi.py (04_simulation)
   is where this gets checked against the real nonlinear system,
   including the severe-saturation test the project summary flags as
   the one that actually exposes anti-windup sign bugs.
"""
import time
import numpy as np
import control
from scipy.signal import StateSpace, step

import genqec_model as gqc
import controls as ctrl
from linearize_condenser import linearize_at_operating_point

# ---- real machine so far: H=15, D=0. Rest of GENQECParams still the
# illustrative placeholder set (see chat) pending the real machine data.
sat = gqc.make_saturation('quadratic', 0.10, 0.30)
P_REAL = gqc.GENQECParams(Ra=0.003, Xl=0.15, Xd=1.80, Xdp=0.30, Xdpp=0.22,
                           Xq=1.70, Xqp=0.50, Xqpp=0.25,
                           Tdop=7.5, Tdopp=0.03, Tqop=0.50, Tqopp=0.05,
                           H=15.0, D=0.0, Kw=0.20, sat=sat, f0=60.0)
AVR = ctrl.AVRParams()  # real: Kp=72,Ki=35,U=[-2.6,3.25],KC=0.0308,FACTOR_MAX=1.4,T_MEAS=0.0111,VREF_MAX=1.15

Rt, Xt, Rline, Xline, Rgrid, Xgrid = 0.002, 0.12, 0.01, 0.08, 0.01, 0.06
Re_total, Xe_total = Rt + Rline + Rgrid, Xt + Xline + Xgrid
R1, X1 = Rt + Rline, Xt + Xline

# real: 8% of condenser nominal, cos(phi)=0.85 lagging, at V=1.0pu
S_LOAD, COSPHI = 0.08, 0.85
P_LOAD, Q_LOAD = S_LOAD * COSPHI, S_LOAD * (1 - COSPHI ** 2) ** 0.5
GLOAD, BLOAD = P_LOAD / 1.0 ** 2, -Q_LOAD / 1.0 ** 2
TDELIV = 0.08

Vt0, P0_OP, Q0_OP = 1.0, 0.0, 0.35   # design point (same illustrative Q0 used throughout)


def get_reduced_plant(order=6, verbose=True):
    """Returns (Ar, Br, Cr, Dr, full_result) -- Qdeliv_m-only output,
    matchdc-truncated to `order`."""
    r = linearize_at_operating_point(P_REAL, AVR, Re_total, Xe_total, R1, X1, Vt0, P0_OP, Q0_OP,
                                      0.10, 0.30, 'quadratic', Gload=GLOAD, Bload=BLOAD, Tdeliv_val=TDELIV)
    A, B, C_full = r['A'], r['B'], r['C']
    q_idx = r['output_names'].index('Qdeliv_m')
    Cq = C_full[q_idx:q_idx + 1, :]
    sys_full = control.ss(A, B, Cq, np.zeros((1, 1)))
    sys_r = control.balred(sys_full, order, method='matchdc')
    if verbose:
        t = np.linspace(0, 3, 1500)
        _, y_full = control.step_response(sys_full, t)
        _, y_r = control.step_response(sys_r, t)
        print(f"[reduce] order {A.shape[0]} -> {order}. Max step-response "
              f"diff over 3s: {np.max(np.abs(y_full - y_r)):.3e}. "
              f"DC gain: full={control.dcgain(sys_full):.6f} reduced={control.dcgain(sys_r):.6f}")
    return sys_r.A, sys_r.B, sys_r.C, sys_r.D, r


def _closed_loop_pi(Ar, Br, Cr, Kp, Ki):
    """PI + integrator augmentation, standard SISO form, for the LINEAR
    reduced plant (no saturation -- see validate_pi.py for that)."""
    A_cl = np.block([[Ar - Kp * Br @ Cr, Ki * Br],
                      [-Cr, np.zeros((1, 1))]])
    B_cl = np.vstack([Kp * Br, np.ones((1, 1))])
    C_cl = np.hstack([Cr, np.zeros((1, 1))])
    return A_cl, B_cl, C_cl


def _step_metrics(Ar, Br, Cr, Kp, Ki, t_max=8.0, npts=4000, band=0.02):
    A_cl, B_cl, C_cl = _closed_loop_pi(Ar, Br, Cr, Kp, Ki)
    eigs = np.linalg.eigvals(A_cl)
    if np.max(eigs.real) >= -1e-6:
        return None
    sys = StateSpace(A_cl, B_cl, C_cl, np.zeros((1, 1)))
    tt, y = step(sys, T=np.linspace(0, t_max, npts))
    yfinal = y[-1]
    if yfinal <= 1e-6:
        return None
    overshoot = (np.max(y) - yfinal) / yfinal
    out_of_band = np.abs(y - yfinal) > band * abs(yfinal)
    t_settle = 0.0 if not np.any(out_of_band) else tt[np.where(out_of_band)[0][-1]]
    return dict(t_settle=t_settle, overshoot=overshoot, eigs=eigs)


def design_pi(Ar, Br, Cr, Kp_range=(0.10, 1.2, 40), Ki_range=(0.5, 6.0, 40), refine=True):
    """Grid search for min settling time (2% band). Two passes: coarse
    over Kp_range/Ki_range, then a finer pass centred on the coarse
    winner (see chat for why: the coarse-grid optimum undersells what's
    achievable -- there's a well-behaved sweet spot a finer grid finds
    that is BOTH faster and lower-overshoot, not a trade needing a cap).
    """
    Kp_grid = np.linspace(*Kp_range)
    Ki_grid = np.linspace(*Ki_range)
    results = []
    for Kp in Kp_grid:
        for Ki in Ki_grid:
            m = _step_metrics(Ar, Br, Cr, Kp, Ki)
            if m is not None:
                results.append((Kp, Ki, m['t_settle'], m['overshoot']))
    best = min(results, key=lambda r: r[2])
    if refine:
        dKp = (Kp_range[1] - Kp_range[0]) / Kp_range[2]
        dKi = (Ki_range[1] - Ki_range[0]) / Ki_range[2]
        Kp_r = (max(best[0] - 3 * dKp, 1e-3), best[0] + 3 * dKp, 40)
        Ki_r = (max(best[1] - 3 * dKi, 1e-3), best[1] + 3 * dKi, 40)
        return design_pi(Ar, Br, Cr, Kp_r, Ki_r, refine=False)
    return best  # (Kp, Ki, t_settle, overshoot)


if __name__ == "__main__":
    t0 = time.time()
    Ar, Br, Cr, Dr, full_result = get_reduced_plant(order=6)
    Kp, Ki, t_settle, overshoot = design_pi(Ar, Br, Cr)
    print(f"\nPI elegido: Kp={Kp:.4f}  Ki={Ki:.4f}")
    print(f"Prediccion en modelo LINEAL reducido: t_settle(2%)={t_settle:.4f}s  overshoot={overshoot*100:.3f}%")
    m = _step_metrics(Ar, Br, Cr, Kp, Ki)
    print("Polos en lazo cerrado (lineal):")
    for e in sorted(m['eigs'], key=lambda z: -z.real):
        print(f"  {e.real:10.5f} + {e.imag:9.5f}j")
    np.savez(_os.path.join(OUTPUTS_DIR, "reduced_plant_qonly.npz"), A=Ar, B=Br, C=Cr, D=Dr)
    np.savez(_os.path.join(OUTPUTS_DIR, "pi_gains.npz"), Kp=Kp, Ki=Ki)
    print(f"\nGuardado en outputs/. Total: {time.time()-t0:.1f}s")
