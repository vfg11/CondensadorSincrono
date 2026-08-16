import sys as _sys, os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ['01_model', '02_linearization', '03_design', '04_simulation']:
    _p = _os.path.join(_PKG_ROOT, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
OUTPUTS_DIR = _os.path.join(_PKG_ROOT, 'outputs')
_os.makedirs(OUTPUTS_DIR, exist_ok=True)
"""
validate_lqi.py
==================
Full NONLINEAR closed loop: 11-state nonlinear plant (same as
validate_pi.py) + 6-state LINEAR Kalman observer (Ar,Br,Cr,L from
design_lqi.py) + integral state xi + LQI control law u = -Kx@xhat -
Ki*xi -> Vref, through the SAME AVR (Kp=72, U_MAX=3.25, VREF_MAX=1.15).

Total simulated states: 11 (plant) + 6 (observer) + 1 (integral) = 18.

Anti-windup: same continuous back-calculation as validate_pi.py, same
reasoning (the discrete formula's sign convention isn't reliably
reconstructable without the original code -- re-derived and verified
here independently instead, same severe-saturation test).

Reference-frame note (see chat/project summary sec.5): xhat0 and xi0
are initialised from the ACTUAL equilibrium at Q0_actual (via
init_plant, exactly like validate_pi.py), not hardcoded to the design
point -- matters once this gets run off-design (battery item 2), a
no-op right now since Q0_actual==Q0_OP in the tests below.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import genqec_model as gqc
import controls as ctrl
from reduce_and_design_pi import (P_REAL as P, AVR, Re_total, Xe_total, R1, X1,
                                   Vt0, P0_OP, Q0_OP, GLOAD, BLOAD, TDELIV)
from validate_pi import init_plant, outputs_raw

KB_BACKCALC_FACTOR = 1.0


def closed_loop_derivatives_lqi(t, z, Pmech0, Einf_t, Ar, Br, Cr, K, L, qref_func, Gf_t, Bf_t,
                                 Vref0, Qbaseline):
    """z = [x_full(11) plant, xhat(6) observer, xi(1) integral]. xhat and
    the reference/measurement seen by the observer are DEVIATION
    variables around Qbaseline (see chat, project summary sec.5 on
    reference-frame bugs) -- Qbaseline must be the REAL equilibrium of
    wherever Q0_actual actually is, not a hardcoded design-point value."""
    x_full = z[:11]
    xhat = z[11:17]
    xi = z[17]

    x6 = x_full[:6]; x_avr = x_full[6:9]; x_deliv = x_full[9:11]
    Gf, Bf = Gf_t(t), Bf_t(t)
    Einf_val = Einf_t(t)
    delta, omega, Eqp, psidp, Edp, psiqp = x6
    Id, Iq, Vd, Vq, Sa, _ = gqc.solve_network(delta, omega, Eqp, psidp, Edp, psiqp,
                                                P, Einf_val, Re_total, Xe_total, Gf=Gf, Bf=Bf)
    Vtgen = float(np.hypot(Vd, Vq))
    Ifd = gqc.field_current(Eqp, psidp, Id, Sa, P)

    y_meas_dev = x_deliv[1] - Qbaseline    # DEVIATION Qdeliv_m -- what the observer compares against
    Qref = qref_func(t)                     # already a deviation (see run_*_test_lqi)

    Kx, Ki = K[0, :6], K[0, 6]
    u_pi = -(Kx @ xhat) - Ki * xi
    Vref_command = Vref0 + u_pi
    Vref_applied = float(np.clip(Vref_command, AVR.VREF_MIN, AVR.VREF_MAX))
    Kb = KB_BACKCALC_FACTOR / abs(Ki) if Ki != 0 else 0.0
    error = Qref - y_meas_dev
    dxi = error + Kb * (Vref_applied - Vref_command)

    dx_avr, Efd = ctrl.avr_derivatives(x_avr, Vref_command, Vtgen, Ifd, AVR)
    dx6 = gqc.derivatives(t, x6, P, Efd, Pmech0, Einf_val, Re_total, Xe_total, Gf=Gf, Bf=Bf)
    V_deliv_raw, Q_deliv_raw = outputs_raw(Vd, Vq, Id, Iq)
    dVdeliv_m = (V_deliv_raw - x_deliv[0]) / TDELIV
    dQdeliv_m = (Q_deliv_raw - x_deliv[1]) / TDELIV

    # observer: LINEAR model of the plant (in deviation variables),
    # corrected by L*(measured deviation - predicted deviation).
    dxhat = Ar @ xhat + Br.flatten() * u_pi + L.flatten() * (y_meas_dev - (Cr @ xhat)[0])

    return np.concatenate([dx6, dx_avr, [dVdeliv_m, dQdeliv_m], dxhat, [dxi]])


def _init_z0(Q0_actual):
    x_full0_pi, Efd0, Pmech0, Einf_val, Vref0, Qbaseline = init_plant(Q0_actual)
    x_full0 = x_full0_pi[:11]   # drop init_plant's trailing PI-only xi_outer placeholder
    xhat0 = np.zeros(6)   # deviation variables: 0 at the true equilibrium, by construction
    xi0 = 0.0
    z0 = np.concatenate([x_full0, xhat0, [xi0]])
    return z0, Efd0, Pmech0, Einf_val, Vref0, Qbaseline


def run_step_test_lqi(Ar, Br, Cr, K, L, Q0_actual, step_size, t_total=10.0, t_event=1.0,
                       method='Radau', max_step=0.02):
    z0, Efd0, Pmech0, Einf_val, Vref0, Qbaseline = _init_z0(Q0_actual)
    # observer/integral run in DEVIATION space (xhat=0 at equilibrium),
    # so the reference is a deviation from Qbaseline too: 0 pre-step,
    # step_size post-step.
    qref_func = lambda t: step_size if t >= t_event else 0.0
    Gf_t = lambda t: GLOAD
    Bf_t = lambda t: BLOAD
    Einf_t = lambda t: Einf_val
    sol = solve_ivp(closed_loop_derivatives_lqi, [0, t_total], z0, method=method,
                     args=(Pmech0, Einf_t, Ar, Br, Cr, K, L, qref_func, Gf_t, Bf_t, Vref0, Qbaseline),
                     max_step=max_step, rtol=1e-7, atol=1e-9, dense_output=True)
    return sol, Qbaseline, Vref0


def run_fault_test_lqi(Ar, Br, Cr, K, L, Q0_actual, fault_duration=0.15, Gfault=25.0,
                        t_total=12.0, t_fault=1.0, max_step=0.005):
    z0, Efd0, Pmech0, Einf_val, Vref0, Qbaseline = _init_z0(Q0_actual)
    qref_func = lambda t: 0.0   # setpoint fixed at baseline (deviation=0 throughout)
    Gf_t = lambda t: (GLOAD + Gfault) if (t_fault <= t < t_fault + fault_duration) else GLOAD
    Bf_t = lambda t: BLOAD
    Einf_t = lambda t: Einf_val
    sol = solve_ivp(closed_loop_derivatives_lqi, [0, t_total], z0, method='Radau',
                     args=(Pmech0, Einf_t, Ar, Br, Cr, K, L, qref_func, Gf_t, Bf_t, Vref0, Qbaseline),
                     max_step=max_step, rtol=1e-7, atol=1e-9, dense_output=True)
    return sol, Qbaseline, Vref0


def run_grid_voltage_test_lqi(Ar, Br, Cr, K, L, Q0_actual, einf_step_frac,
                               t_total=8.0, t_event=1.0, max_step=0.02):
    """Grid voltage disturbance, LQI version -- see run_grid_voltage_test
    (validate_pi.py) for the rationale. Q_ref held fixed (deviation=0)."""
    z0, Efd0, Pmech0, Einf_val, Vref0, Qbaseline = _init_z0(Q0_actual)
    qref_func = lambda t: 0.0
    Gf_t = lambda t: GLOAD
    Bf_t = lambda t: BLOAD
    Einf_t = lambda t: Einf_val * (1.0 + einf_step_frac) if t >= t_event else Einf_val
    sol = solve_ivp(closed_loop_derivatives_lqi, [0, t_total], z0, method='Radau',
                     args=(Pmech0, Einf_t, Ar, Br, Cr, K, L, qref_func, Gf_t, Bf_t, Vref0, Qbaseline),
                     max_step=max_step, rtol=1e-7, atol=1e-9, dense_output=True)
    return sol, Qbaseline, Vref0


def _metrics(sol, Qbaseline, step_size, t_event=1.0):
    t = sol.t
    Qdm = sol.y[10, :]
    post = t >= t_event
    yfinal = Qdm[-1]
    band = 0.02 * abs(step_size) if step_size != 0 else 0.02 * abs(yfinal - Qbaseline)
    ref = Qbaseline + step_size
    out_of_band = np.abs(Qdm[post] - ref) > band
    t_settle = 0.0 if not np.any(out_of_band) else t[post][np.where(out_of_band)[0][-1]] - t_event
    overshoot = (np.max(Qdm[post]) - ref) / step_size if step_size != 0 else 0.0
    return t_settle, overshoot


def refine_lqi_nonlinear(qy_grid, qi_grid, Q0_actual=None, step_size=0.15, overshoot_cap=0.10,
                          eps=1e-3, r=1.0, observer_speedup=4.0):
    """Same lesson as the PI (see chat/validate_pi.py): the linear-model
    LQR optimum saturates the AVR's own PI almost immediately. Search
    directly against the nonlinear closed loop instead."""
    from design_lqi import build_augmented, lqr_gain, kalman_gain, pick_observer_q_proc, _closed_loop_poles
    from reduce_and_design_pi import get_reduced_plant, Q0_OP
    Q0_actual = Q0_OP if Q0_actual is None else Q0_actual
    Ar, Br, Cr, Dr, _ = get_reduced_plant(order=6, verbose=False)
    Aaug, Baug = build_augmented(Ar, Br, Cr)

    best = None
    for qy in qy_grid:
        for qi in qi_grid:
            K, _, _ = lqr_gain(Aaug, Baug, Cr, qy, qi, eps, r)
            ctrl_slowest = np.max(_closed_loop_poles(Aaug, Baug, K).real)
            if ctrl_slowest >= -1e-6:
                continue
            q_proc = pick_observer_q_proc(Ar, Cr, Br, ctrl_slowest * observer_speedup)
            L = kalman_gain(Ar, Br, Cr, q_proc, 1.0)
            sol, Qbase, _ = run_step_test_lqi(Ar, Br, Cr, K, L, Q0_actual, step_size,
                                               t_total=6.0, t_event=1.0)
            t_settle, overshoot = _metrics(sol, Qbase, step_size)
            if overshoot > overshoot_cap:
                continue
            if best is None or t_settle < best[2]:
                best = (qy, qi, t_settle, overshoot, K, L, q_proc)
    return best


if __name__ == "__main__":
    from design_lqi import build_augmented, lqr_gain, kalman_gain, pick_observer_q_proc, _closed_loop_poles
    from reduce_and_design_pi import get_reduced_plant

    Ar, Br, Cr, Dr, _ = get_reduced_plant(order=6, verbose=False)
    Aaug, Baug = build_augmented(Ar, Br, Cr)

    def build_K_L(qy, qi, eps=1e-3, r=1.0, speedup=4.0):
        K, _, _ = lqr_gain(Aaug, Baug, Cr, qy, qi, eps, r)
        ctrl_slowest = np.max(_closed_loop_poles(Aaug, Baug, K).real)
        q_proc = pick_observer_q_proc(Ar, Cr, Br, ctrl_slowest * speedup)
        L = kalman_gain(Ar, Br, Cr, q_proc, 1.0)
        return K, L

    def _vref_trace(sol, Vref0, K):
        xhat = sol.y[11:17, :]
        xi = sol.y[17, :]
        Kx, Ki = K[0, :6], K[0, 6]
        u_pi = -(Kx @ xhat) - Ki * xi
        Vref_cmd = Vref0 + u_pi
        return np.minimum(Vref_cmd, AVR.VREF_MAX), xi

    print("=== LQI 'optimo lineal' (qy=2.848, qi=1000) contra el modelo NO lineal ===")
    K_lin, L_lin = build_K_L(2.848, 1000.0)
    sol1a, Qbase1a, Vref0_1a = run_step_test_lqi(Ar, Br, Cr, K_lin, L_lin, Q0_OP, 0.15,
                                                  t_total=6.0, t_event=1.0)
    ts1a, ov1a = _metrics(sol1a, Qbase1a, 0.15)
    Vref1a, _ = _vref_trace(sol1a, Vref0_1a, K_lin)
    print(f"  NO LINEAL: t_settle={ts1a:.3f}s  overshoot={ov1a*100:.2f}%  "
          f"Vref max={np.max(Vref1a):.4f} (limite {AVR.VREF_MAX})  <<< satura, mismo problema que el PI\n")

    print("=== LQI final (qy=1.4, qi=20 -- refinado contra el modelo no lineal) ===")
    K, L = build_K_L(1.4, 20.0)
    print(f"K={K.flatten()}\nL={L.flatten()}\n")

    print("--- Test 1: escalon +0.15 pu en Q_ref ---")
    sol1, Qbase1, Vref0_1 = run_step_test_lqi(Ar, Br, Cr, K, L, Q0_OP, 0.15, t_total=6.0, t_event=1.0)
    ts1, ov1 = _metrics(sol1, Qbase1, 0.15)
    Vref1, xi1 = _vref_trace(sol1, Vref0_1, K)
    print(f"  t_settle={ts1:.3f}s  overshoot={ov1*100:.2f}%  Vref max={np.max(Vref1):.4f}")
    print(f"  (PI equivalente: t_settle=1.080s overshoot=1.61%)\n")

    print("--- Test 2: cortocircuito trifasico 150ms ---")
    sol2, Qbase2, Vref0_2 = run_fault_test_lqi(Ar, Br, Cr, K, L, Q0_OP, fault_duration=0.15,
                                                t_fault=1.0, t_total=12.0)
    Vref2, xi2 = _vref_trace(sol2, Vref0_2, K)
    Qdm2 = sol2.y[10, :]
    print(f"  Vref durante/post-falla: max={np.max(Vref2):.4f}  min={np.min(Vref2):.4f}  "
          f"(limite {AVR.VREF_MAX})")
    print(f"  xi (integral): max_abs={np.max(np.abs(xi2)):.4f}  final(t=12s)={xi2[-1]:.2e}")
    print(f"  Q_deliv_m: min durante falla={np.min(Qdm2):.4f}  final(t=12s)={Qdm2[-1]:.4f}  "
          f"objetivo={Qbase2:.4f}  error={abs(Qdm2[-1]-Qbase2):.2e}")
    recovered = abs(Qdm2[-1] - Qbase2) < 5e-3 and abs(xi2[-1]) < 0.1
    print(f"  Recuperacion limpia: {'SI' if recovered else 'NO -- revisar'}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(sol1.t, sol1.y[10, :]); axes[0, 0].axhline(Qbase1 + 0.15, ls='--', c='gray')
    axes[0, 0].set_title(f'LQI Test 1: Q_deliv_m (escalon +0.15)'); axes[0, 0].set_xlabel('t [s]')
    axes[0, 1].plot(sol1.t, Vref1); axes[0, 1].axhline(AVR.VREF_MAX, ls='--', c='r')
    axes[0, 1].set_title('LQI Test 1: Vref aplicado'); axes[0, 1].set_xlabel('t [s]')
    axes[1, 0].plot(sol2.t, Qdm2); axes[1, 0].axhline(Qbase2, ls='--', c='gray')
    axes[1, 0].set_title('LQI Test 2: Q_deliv_m (falla 150ms)'); axes[1, 0].set_xlabel('t [s]')
    axes[1, 1].plot(sol2.t, Vref2, label='Vref aplicado')
    axes[1, 1].plot(sol2.t, xi2, label='xi (integral)')
    axes[1, 1].axhline(AVR.VREF_MAX, ls='--', c='r')
    axes[1, 1].set_title('LQI Test 2: Vref y estado integral'); axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_xlabel('t [s]')
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUTS_DIR, 'validate_lqi.png'), dpi=110)
    np.savez(_os.path.join(OUTPUTS_DIR, 'lqi_gains_final.npz'), K=K, L=L, qy=1.4, qi=20.0,
             qy_linear_optimal=2.848, qi_linear_optimal=1000.0)
    print(f"\nGuardado outputs/validate_lqi.png y outputs/lqi_gains_final.npz")

