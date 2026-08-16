import sys as _sys, os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ['01_model', '02_linearization', '03_design', '04_simulation']:
    _p = _os.path.join(_PKG_ROOT, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
OUTPUTS_DIR = _os.path.join(_PKG_ROOT, 'outputs')
_os.makedirs(OUTPUTS_DIR, exist_ok=True)
"""
validate_pi.py
=================
Full NONLINEAR closed loop: machine (genqec_model) + AVR (controls.py,
its own internal PI+FEX+VREF_MAX clamp) + outer PI on Qdeliv_m (the
LAGGED/measured signal, not the instantaneous algebraic Q_delivered) +
the two delivery-point measurement lags -- 12 states total.

Outer PI gains (Kp, Ki) come from reduce_and_design_pi.py (grid search
on the order-6 reduced LINEAR model). This script is where that design
actually gets tested against reality: saturation (VREF_MAX=1.15),
anti-windup, and the nonlinear plant the linear design never saw.

Anti-windup: standard continuous back-calculation, dxi/dt = error +
Kb*(Vref_applied - Vref_command), Kb=1/Ki -- NOT the previous project's
exact discrete formula (see chat: that formula's sign convention isn't
reliably reconstructable from the summary alone without the original
code, so this was re-derived from first principles instead and is
verified here, independently, the same way the summary recommends: a
severe, sustained saturation test, not just gentle steps).
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

KB_BACKCALC_FACTOR = 1.0  # Kb = KB_BACKCALC_FACTOR / Ki_outer


def init_plant(Q0_actual):
    """Consistent equilibrium at Q0_actual (machine's own terminal Q,
    load+delivery lags included). Returns everything the RHS needs."""
    state0, Efd0, Pmech0, Einf_val = gqc.initialize(P, Vt0, P0_OP, Q0_actual, Re_total, Xe_total,
                                                      Gf=GLOAD, Bf=BLOAD)
    Id0, Iq0, Vd0, Vq0, Sa0, _ = gqc.solve_network(*state0, P, Einf_val, Re_total, Xe_total,
                                                     Gf=GLOAD, Bf=BLOAD)
    Vtgen0 = float(np.hypot(Vd0, Vq0))
    Ifd0 = gqc.field_current(state0[2], state0[3], Id0, Sa0, P)
    avr_x0, Vref0 = ctrl.avr_initialize(Vtgen0, Efd0, Ifd0, AVR)

    Id_load0 = GLOAD * Vd0 - BLOAD * Vq0
    Iq_load0 = GLOAD * Vq0 + BLOAD * Vd0
    Id_net0, Iq_net0 = Id0 - Id_load0, Iq0 - Iq_load0
    Vd_deliv0 = Vd0 - R1 * Id_net0 + X1 * Iq_net0
    Vq_deliv0 = Vq0 - R1 * Iq_net0 - X1 * Id_net0
    Vdeliv_raw0 = float(np.hypot(Vd_deliv0, Vq_deliv0))
    Qdeliv_raw0 = float(Vq_deliv0 * Id_net0 - Vd_deliv0 * Iq_net0)

    x_full0 = np.concatenate([state0, avr_x0, [Vdeliv_raw0, Qdeliv_raw0], [0.0]])
    return x_full0, Efd0, Pmech0, Einf_val, Vref0, Qdeliv_raw0


def outputs_raw(Vd, Vq, Id, Iq):
    Id_load = GLOAD * Vd - BLOAD * Vq
    Iq_load = GLOAD * Vq + BLOAD * Vd
    Id_net, Iq_net = Id - Id_load, Iq - Iq_load
    Vd_deliv = Vd - R1 * Id_net + X1 * Iq_net
    Vq_deliv = Vq - R1 * Iq_net - X1 * Id_net
    V_deliv = np.hypot(Vd_deliv, Vq_deliv)
    Q_deliv = Vq_deliv * Id_net - Vd_deliv * Iq_net
    return V_deliv, Q_deliv


def closed_loop_derivatives(t, x_full, Pmech0, Einf_t, Kp, Ki, qref_func, Gf_t, Bf_t, Vref0):
    x6 = x_full[:6]
    x_avr = x_full[6:9]
    x_deliv = x_full[9:11]
    xi_outer = x_full[11]

    Gf, Bf = Gf_t(t), Bf_t(t)
    Einf_val = Einf_t(t)
    delta, omega, Eqp, psidp, Edp, psiqp = x6
    Id, Iq, Vd, Vq, Sa, _ = gqc.solve_network(delta, omega, Eqp, psidp, Edp, psiqp,
                                                P, Einf_val, Re_total, Xe_total, Gf=Gf, Bf=Bf)
    Vtgen = float(np.hypot(Vd, Vq))
    Ifd = gqc.field_current(Eqp, psidp, Id, Sa, P)

    Qref = qref_func(t)
    error = Qref - x_deliv[1]           # feedback = Qdeliv_m (measured/lagged), not raw
    u_pi = Kp * error + Ki * xi_outer
    Vref_command = Vref0 + u_pi
    Vref_applied = float(np.clip(Vref_command, AVR.VREF_MIN, AVR.VREF_MAX))
    Kb = KB_BACKCALC_FACTOR / Ki
    dxi_outer = error + Kb * (Vref_applied - Vref_command)

    dx_avr, Efd = ctrl.avr_derivatives(x_avr, Vref_command, Vtgen, Ifd, AVR)
    dx6 = gqc.derivatives(t, x6, P, Efd, Pmech0, Einf_val, Re_total, Xe_total, Gf=Gf, Bf=Bf)

    V_deliv_raw, Q_deliv_raw = outputs_raw(Vd, Vq, Id, Iq)
    dVdeliv_m = (V_deliv_raw - x_deliv[0]) / TDELIV
    dQdeliv_m = (Q_deliv_raw - x_deliv[1]) / TDELIV

    return np.concatenate([dx6, dx_avr, [dVdeliv_m, dQdeliv_m], [dxi_outer]])


def run_step_test(Kp, Ki, Q0_actual, step_size, t_total=10.0, t_event=1.0):
    x_full0, Efd0, Pmech0, Einf_val, Vref0, Qbaseline = init_plant(Q0_actual)
    qref_func = lambda t: Qbaseline + (step_size if t >= t_event else 0.0)
    Gf_t = lambda t: GLOAD
    Bf_t = lambda t: BLOAD
    Einf_t = lambda t: Einf_val
    sol = solve_ivp(closed_loop_derivatives, [0, t_total], x_full0, method='Radau',
                     args=(Pmech0, Einf_t, Kp, Ki, qref_func, Gf_t, Bf_t, Vref0),
                     max_step=0.02, rtol=1e-8, atol=1e-10, dense_output=True)
    return sol, Qbaseline, Vref0


def run_fault_test(Kp, Ki, Q0_actual, fault_duration=0.15, Gfault=25.0,
                    t_total=8.0, t_fault=1.0):
    """Three-phase short circuit AT THE DELIVERY POINT approximated as a
    heavy shunt fault at the machine terminal (Gfault >> Gload) added on
    top of the permanent load admittance for fault_duration, then
    cleared (back to Gload,Bload only) -- no reclosing."""
    x_full0, Efd0, Pmech0, Einf_val, Vref0, Qbaseline = init_plant(Q0_actual)
    qref_func = lambda t: Qbaseline   # setpoint held fixed -- pure disturbance rejection
    Gf_t = lambda t: (GLOAD + Gfault) if (t_fault <= t < t_fault + fault_duration) else GLOAD
    Bf_t = lambda t: BLOAD
    Einf_t = lambda t: Einf_val
    sol = solve_ivp(closed_loop_derivatives, [0, t_total], x_full0, method='Radau',
                     args=(Pmech0, Einf_t, Kp, Ki, qref_func, Gf_t, Bf_t, Vref0),
                     max_step=0.005, rtol=1e-8, atol=1e-10, dense_output=True)
    return sol, Qbaseline, Vref0


def run_grid_voltage_test(Kp, Ki, Q0_actual, einf_step_frac, t_total=8.0, t_event=1.0):
    """Grid voltage disturbance (project summary battery item 4): Einf
    receives a step of einf_step_frac (e.g. +0.05 = +5%), Q_ref held
    FIXED -- pure external-disturbance rejection, separate from the
    setpoint-tracking steps above."""
    x_full0, Efd0, Pmech0, Einf_val, Vref0, Qbaseline = init_plant(Q0_actual)
    qref_func = lambda t: Qbaseline
    Gf_t = lambda t: GLOAD
    Bf_t = lambda t: BLOAD
    Einf_t = lambda t: Einf_val * (1.0 + einf_step_frac) if t >= t_event else Einf_val
    sol = solve_ivp(closed_loop_derivatives, [0, t_total], x_full0, method='Radau',
                     args=(Pmech0, Einf_t, Kp, Ki, qref_func, Gf_t, Bf_t, Vref0),
                     max_step=0.02, rtol=1e-8, atol=1e-10, dense_output=True)
    return sol, Qbaseline, Vref0


def _recover_signals(sol, Vref0, Kp, Ki, qref_func):
    t = sol.t
    x = sol.y
    Qdeliv_m = x[10, :]
    Vdeliv_m = x[9, :]
    xi_outer = x[11, :]
    error = np.array([qref_func(tt) for tt in t]) - Qdeliv_m
    u_pi = Kp * error + Ki * xi_outer
    Vref_command = Vref0 + u_pi
    Vref_applied = np.clip(Vref_command, AVR.VREF_MIN, AVR.VREF_MAX)
    return dict(t=t, Qdeliv_m=Qdeliv_m, Vdeliv_m=Vdeliv_m, xi_outer=xi_outer,
                Vref_command=Vref_command, Vref_applied=Vref_applied)


def _step_settle_overshoot(sol, Qbase, step_size, t_event=1.0):
    t = sol.t
    Qdm = sol.y[10, :]
    post = t >= t_event
    yfinal = Qdm[-1]
    band = 0.02 * abs(yfinal - Qbase)
    out_of_band = np.abs(Qdm[post] - yfinal) > band
    t_settle = 0.0 if not np.any(out_of_band) else t[post][np.where(out_of_band)[0][-1]] - t_event
    overshoot = (np.max(Qdm[post]) - yfinal) / (yfinal - Qbase) if (yfinal - Qbase) != 0 else 0.0
    return t_settle, overshoot


def refine_pi_nonlinear(Kp_grid, Ki_grid, Q0_actual=None, step_size=0.15, overshoot_cap=0.10):
    """Grid search directly on the FULL NONLINEAR closed loop -- see
    module docstring / chat: the linear-model optimum (reduce_and_
    design_pi.design_pi) turned out to saturate the AVR's own internal
    PI almost immediately (Kp_avr=72, U_MAX=3.25 -> only ~0.045pu of Vt
    error before it clips -- see chat), giving 2.0s/33.6% overshoot in
    reality against a linear prediction of 0.34s/1.9%. This searches
    where the linear search can't see: directly against the saturating,
    nonlinear plant."""
    from reduce_and_design_pi import Q0_OP
    Q0_actual = Q0_OP if Q0_actual is None else Q0_actual
    best = None
    for Kp in Kp_grid:
        for Ki in Ki_grid:
            sol, Qbase, _ = run_step_test(Kp, Ki, Q0_actual, step_size, t_total=6.0, t_event=1.0)
            t_settle, overshoot = _step_settle_overshoot(sol, Qbase, step_size)
            if overshoot > overshoot_cap:
                continue
            if best is None or t_settle < best[2]:
                best = (Kp, Ki, t_settle, overshoot)
    return best


if __name__ == "__main__":
    from reduce_and_design_pi import get_reduced_plant, design_pi
    Ar, Br, Cr, Dr, _ = get_reduced_plant(order=6, verbose=False)
    Kp_lin, Ki_lin, t_settle_lin, overshoot_lin = design_pi(Ar, Br, Cr)
    print(f"PI 'optimo' del modelo LINEAL: Kp={Kp_lin:.4f} Ki={Ki_lin:.4f}  "
          f"(prediccion: t_settle={t_settle_lin:.3f}s, overshoot={overshoot_lin*100:.2f}%)\n")

    print("=== Test 1a: escalon +0.15 pu con el PI 'optimo lineal' (referencia) ===")
    sol1a, Qbase1a, _ = run_step_test(Kp_lin, Ki_lin, Q0_OP, step_size=0.15, t_total=6.0, t_event=1.0)
    ts1a, ov1a = _step_settle_overshoot(sol1a, Qbase1a, 0.15)
    print(f"  NO LINEAL: t_settle={ts1a:.3f}s  overshoot={ov1a*100:.2f}%  <<< muy por debajo de lo previsto")
    print(f"  Causa (ver chat): u_pi interno del AVR satura casi de inmediato "
          f"(Kp_avr=72, U_MAX=3.25 -> ~0.045pu de margen en Vt)\n")

    print("=== Refinando Kp,Ki directamente contra el modelo NO LINEAL ===")
    # Ki=0.7 excluido: esa franja resulta numericamente patologica para
    # el integrador (Radau se atasca, independientemente de Kp -- ver
    # chat) en varios puntos de la rejilla; no aporta nada que Ki=0.6 no
    # cubra ya, así que se evita en vez de perseguir la causa exacta.
    Kp, Ki, ts_ref, ov_ref = refine_pi_nonlinear(
        Kp_grid=[0.05, 0.06, 0.07, 0.08, 0.09],
        Ki_grid=[0.4, 0.5, 0.6], overshoot_cap=0.10)
    print(f"  PI final: Kp={Kp:.4f} Ki={Ki:.4f}  (t_settle={ts_ref:.3f}s, overshoot={ov_ref*100:.2f}%, "
          f"NO LINEAL de verdad, no prediccion)\n")

    print("=== Test 1: escalon moderado (+0.15 pu en Q_ref), PI final ===")
    sol1, Qbase1, Vref0_1 = run_step_test(Kp, Ki, Q0_OP, step_size=0.15, t_total=6.0, t_event=1.0)
    qref1 = lambda t: Qbase1 + (0.15 if t >= 1.0 else 0.0)
    sig1 = _recover_signals(sol1, Vref0_1, Kp, Ki, qref1)
    t_settle_nl, overshoot_nl = _step_settle_overshoot(sol1, Qbase1, 0.15)
    print(f"  t_settle={t_settle_nl:.3f}s  overshoot={overshoot_nl*100:.2f}%  "
          f"Vref max={np.max(sig1['Vref_applied']):.4f} (VREF_MAX={AVR.VREF_MAX})")

    print("\n=== Test 2: cortocircuito trifasico 150ms (saturacion severa), PI final ===")
    sol2, Qbase2, Vref0_2 = run_fault_test(Kp, Ki, Q0_OP, fault_duration=0.15, t_fault=1.0, t_total=12.0)
    qref2 = lambda t: Qbase2
    sig2 = _recover_signals(sol2, Vref0_2, Kp, Ki, qref2)
    print(f"  Vref durante/post-falla: max={np.max(sig2['Vref_applied']):.4f}  "
          f"min={np.min(sig2['Vref_applied']):.4f}  (VREF_MAX={AVR.VREF_MAX})")
    print(f"  xi_outer (integral): max_abs={np.max(np.abs(sig2['xi_outer'])):.4f}  "
          f"final(t=12s)={sig2['xi_outer'][-1]:.5f}  (debe volver cerca de 0, no divergir)")
    Qfinal2 = sig2['Qdeliv_m'][-1]
    print(f"  Q_deliv_m: min durante falla={np.min(sig2['Qdeliv_m']):.4f}  "
          f"final(t=12s)={Qfinal2:.4f}  (objetivo={Qbase2:.4f}, error={abs(Qfinal2-Qbase2):.2e})")
    recovered = abs(Qfinal2 - Qbase2) < 5e-3 and abs(sig2['xi_outer'][-1]) < 0.1
    print(f"  Recuperacion limpia (sin windup): {'SI' if recovered else 'NO -- revisar signo antiwindup'}")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(sig1['t'], sig1['Qdeliv_m']); axes[0, 0].axhline(Qbase1 + 0.15, ls='--', c='gray')
    axes[0, 0].set_title(f'Test 1: Q_deliv_m (escalon +0.15) -- Kp={Kp:.3f} Ki={Ki:.3f}')
    axes[0, 0].set_xlabel('t [s]')
    axes[0, 1].plot(sig1['t'], sig1['Vref_applied']); axes[0, 1].axhline(AVR.VREF_MAX, ls='--', c='r')
    axes[0, 1].set_title('Test 1: Vref aplicado'); axes[0, 1].set_xlabel('t [s]')
    axes[1, 0].plot(sig2['t'], sig2['Qdeliv_m']); axes[1, 0].axhline(Qbase2, ls='--', c='gray')
    axes[1, 0].set_title('Test 2: Q_deliv_m (falla 150ms)'); axes[1, 0].set_xlabel('t [s]')
    axes[1, 1].plot(sig2['t'], sig2['Vref_applied'], label='Vref aplicado')
    axes[1, 1].plot(sig2['t'], sig2['xi_outer'], label='xi_outer (integral)')
    axes[1, 1].axhline(AVR.VREF_MAX, ls='--', c='r')
    axes[1, 1].set_title('Test 2: Vref y estado integral'); axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_xlabel('t [s]')
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUTS_DIR, 'validate_pi.png'), dpi=110)
    np.savez(_os.path.join(OUTPUTS_DIR, 'pi_gains_final.npz'), Kp=Kp, Ki=Ki,
             Kp_linear_optimal=Kp_lin, Ki_linear_optimal=Ki_lin)
    print(f"\nGuardado outputs/validate_pi.png y outputs/pi_gains_final.npz")

