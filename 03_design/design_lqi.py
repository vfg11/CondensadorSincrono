import sys as _sys, os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ['01_model', '02_linearization', '03_design', '04_simulation']:
    _p = _os.path.join(_PKG_ROOT, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
OUTPUTS_DIR = _os.path.join(_PKG_ROOT, 'outputs')
_os.makedirs(OUTPUTS_DIR, exist_ok=True)
"""
design_lqi.py
================
LQI = LQR (full state feedback on the order-6 reduced model, output
Qdeliv_m -- same reduction as the PI) + integral state on the reference
error + Kalman observer (since only Qdeliv_m is measured, not the 6
reduced states directly -- they are a BALANCED-basis combination of the
original 11, not physically interpretable one-by-one).

Augmented state z=[x(6), xi(1)], xi integrates (r - Cr@x):
    Aaug = [[Ar, 0], [-Cr, 0]]     Baug = [[Br], [0]]     (7x7, 7x1)
Control law: u = -Kx@xhat - Ki*xi = -K@[xhat; xi], K from LQR on
(Aaug, Baug, Qw, Rw).

Weight recipe (mirrors the previous project's Qy/integral-weight/R
split, adapted to a single output): cost = qy*(Cr@x)^2 + qi*xi^2 +
eps*||x||^2 + r*u^2 -- qy weights OUTPUT deviation directly (not the
balanced states, which aren't physically meaningful individually), qi
drives integral/steady-state error down, eps is a small regularizer
for a well-posed Riccati solve, r trades off control effort.

Given what happened with the PI (linear-optimal gains saturated the
AVR's own inner PI almost immediately -- see validate_pi.py/chat), the
SAME two-stage approach is used here: a linear sweep first (fast, gives
an upper bound on achievable aggressiveness and a starting point), then
mandatory verification/refinement against the full NONLINEAR closed
loop in validate_lqi.py -- the linear number is a hypothesis, not the
answer, until nonlinear-checked.

Kalman observer: process noise entering through the input channel
(Qn = q_proc*Br@Br.T, a standard "actuator/model uncertainty" choice
when there's no better-characterized disturbance model), measurement
noise Rn (scalar, on Qdeliv_m). q_proc/Rn ratio set so the observer's
slowest pole is ~4x faster than the controller's slowest closed-loop
pole -- fast enough that estimation error settles well before the
controlled response does, without being needlessly aggressive.
"""
import numpy as np
from scipy.linalg import solve_continuous_are

from reduce_and_design_pi import get_reduced_plant, Q0_OP


def build_augmented(Ar, Br, Cr):
    n = Ar.shape[0]
    Aaug = np.block([[Ar, np.zeros((n, 1))],
                      [-Cr, np.zeros((1, 1))]])
    Baug = np.vstack([Br, np.zeros((1, 1))])
    return Aaug, Baug


def lqr_gain(Aaug, Baug, Cr, qy, qi, eps, r):
    n = Aaug.shape[0] - 1
    Cr_pad = np.hstack([Cr, np.zeros((1, 1))])
    Qw = qy * (Cr_pad.T @ Cr_pad) + np.diag(np.append(eps * np.ones(n), qi))
    Rw = np.array([[r]])
    Pare = solve_continuous_are(Aaug, Baug, Qw, Rw)
    K = np.linalg.solve(Rw, Baug.T @ Pare)   # 1 x (n+1)
    return K, Qw, Rw


def kalman_gain(Ar, Br, Cr, q_proc, r_meas):
    Qn = q_proc * (Br @ Br.T)
    Rn = np.array([[r_meas]])
    Pobs = solve_continuous_are(Ar.T, Cr.T, Qn, Rn)
    L = Pobs @ Cr.T / r_meas   # n x 1
    return L


def _closed_loop_poles(Aaug, Baug, K):
    return np.linalg.eigvals(Aaug - Baug @ K)


def pick_observer_q_proc(Ar, Cr, Br, target_slowest, q_range=None):
    """Direct scan (not bisection -- the max-real-part-vs-q_proc relation
    isn't reliably monotonic once different eigenvalues trade off being
    the 'slowest', which broke a bisection search here; see chat) over
    log-spaced q_proc, returns the one whose observer max-real-part is
    closest to target_slowest without being SLOWER than the controller."""
    if q_range is None:
        q_range = np.logspace(-4, 4, 60)
    best = None
    for q in q_range:
        L = kalman_gain(Ar, Br, Cr, q, 1.0)
        slowest = np.max(np.linalg.eigvals(Ar - L @ Cr).real)
        if slowest >= target_slowest:
            continue   # not fast enough relative to the controller
        if best is None or abs(slowest - target_slowest) < abs(best[1] - target_slowest):
            best = (q, slowest)
    return best[0] if best else q_range[-1]


def _step_metrics_lin(Aaug, Baug, K, t_max=8.0, npts=3000, band=0.02):
    from scipy.signal import StateSpace, step
    Acl = Aaug - Baug @ K
    n = Acl.shape[0]
    Cout = np.zeros((1, n)); Cout[0, -1] = 0  # placeholder, filled by caller via closure
    return Acl


def lqr_step_metrics(Aaug, Baug, Cr, K, t_max=8.0, npts=3000, band=0.02):
    from scipy.signal import StateSpace, step
    n = Aaug.shape[0]
    Acl = Aaug - Baug @ K
    if np.max(np.linalg.eigvals(Acl).real) >= -1e-6:
        return None
    # reference r enters as +1 on the integrator row (see build_augmented);
    # output = Cr @ x (first n-1 states)
    Bcl = np.zeros((n, 1)); Bcl[-1, 0] = 1.0
    Ccl = np.hstack([Cr, np.zeros((1, 1))])
    sys = StateSpace(Acl, Bcl, Ccl, np.zeros((1, 1)))
    tt, y = step(sys, T=np.linspace(0, t_max, npts))
    yfinal = y[-1]
    if yfinal <= 1e-6:
        return None
    overshoot = (np.max(y) - yfinal) / yfinal
    out_of_band = np.abs(y - yfinal) > band * abs(yfinal)
    t_settle = 0.0 if not np.any(out_of_band) else tt[np.where(out_of_band)[0][-1]]
    return dict(t_settle=t_settle, overshoot=overshoot)


def sweep_lqr_weights(Aaug, Baug, Cr, qy_grid, qi_grid, eps=1e-3, r=1.0, overshoot_cap=0.15):
    """Same spirit as the PI's grid search: minimise 2%-band settling
    time on the LINEAR augmented closed loop. Starting point only --
    see chat, this needs nonlinear verification before trusting it."""
    best = None
    for qy in qy_grid:
        for qi in qi_grid:
            K, _, _ = lqr_gain(Aaug, Baug, Cr, qy, qi, eps, r)
            m = lqr_step_metrics(Aaug, Baug, Cr, K)
            if m is None or m['overshoot'] > overshoot_cap:
                continue
            if best is None or m['t_settle'] < best[2]:
                best = (qy, qi, m['t_settle'], m['overshoot'])
    return best


def design(qy=1.0, qi=1.0, eps=1e-3, r=1.0, observer_speedup=4.0, verbose=True):
    Ar, Br, Cr, Dr, _ = get_reduced_plant(order=6, verbose=False)
    Aaug, Baug = build_augmented(Ar, Br, Cr)
    K, Qw, Rw = lqr_gain(Aaug, Baug, Cr, qy, qi, eps, r)
    cl_poles = _closed_loop_poles(Aaug, Baug, K)
    ctrl_slowest = np.max(cl_poles.real)

    target = ctrl_slowest * observer_speedup
    q_proc = pick_observer_q_proc(Ar, Cr, Br, target)
    L = kalman_gain(Ar, Br, Cr, q_proc, 1.0)
    obs_poles = np.linalg.eigvals(Ar - L @ Cr)

    if verbose:
        print(f"K = {K.flatten()}")
        print("Polos en lazo cerrado (controlador, LQR ideal con estado completo):")
        for e in sorted(cl_poles, key=lambda z: -z.real):
            print(f"  {e.real:10.5f} + {e.imag:9.5f}j")
        print(f"\nq_proc(observador)={q_proc:.4e}  (Rn=1.0 fijo)")
        print("Polos del observador (Ar - L*Cr):")
        for e in sorted(obs_poles, key=lambda z: -z.real):
            print(f"  {e.real:10.5f} + {e.imag:9.5f}j")
    return dict(Ar=Ar, Br=Br, Cr=Cr, Aaug=Aaug, Baug=Baug, K=K, L=L,
                cl_poles=cl_poles, obs_poles=obs_poles, q_proc=q_proc,
                qy=qy, qi=qi, eps=eps, r=r)


if __name__ == "__main__":
    Ar, Br, Cr, Dr, _ = get_reduced_plant(order=6, verbose=False)
    Aaug, Baug = build_augmented(Ar, Br, Cr)

    print("=== Barrido de pesos LQR (modelo lineal, igual que se hizo con el PI) ===")
    best = sweep_lqr_weights(Aaug, Baug, Cr,
                              qy_grid=np.logspace(-1, 3, 12),
                              qi_grid=np.logspace(-1, 3, 12), overshoot_cap=0.15)
    print(f"Mejor (lineal): qy={best[0]:.3f} qi={best[1]:.3f} "
          f"t_settle={best[2]:.4f}s overshoot={best[3]*100:.2f}%\n")

    print("=== Diseno completo con esos pesos ===\n")
    d = design(qy=best[0], qi=best[1], eps=1e-3, r=1.0, observer_speedup=4.0)
    np.savez(_os.path.join(OUTPUTS_DIR, "lqi_design.npz"),
             Ar=d['Ar'], Br=d['Br'], Cr=d['Cr'], K=d['K'], L=d['L'],
             qy=d['qy'], qi=d['qi'], eps=d['eps'], r=d['r'], q_proc=d['q_proc'])
    print("\nGuardado outputs/lqi_design.npz")
