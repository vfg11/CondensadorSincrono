import sys as _sys, os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ['01_model', '02_linearization', '03_design', '04_simulation']:
    _p = _os.path.join(_PKG_ROOT, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
OUTPUTS_DIR = _os.path.join(_PKG_ROOT, 'outputs')
_os.makedirs(OUTPUTS_DIR, exist_ok=True)
"""
battery_offdesign_and_grid.py
================================
Items 2 and 4 of the project summary's recommended validation battery
(section 8) -- items 1 and 3 (design-point step, fault) are in
validate_pi.py/validate_lqi.py already.

Item 2: setpoint step at a couple of points MODERATELY AWAY from the
Q0=0.35 design point -- same FIXED gains (Kp/Ki for PI, K/L for LQI),
NOT redesigned. This is the actual test of whether a single design
point generalises, which is the whole premise of this (non-gain-
scheduled) project. Q0 in {0.0, 0.6}: both comfortably inside the
avr_initialize-reachable range (~[-0.6, 0.7], see chat), moderately far
from 0.35 in opposite directions.

Item 4: grid voltage (Einf) step, Q_ref held FIXED -- pure external-
disturbance rejection, at the design point. +-5% and +-10%.

Reference-frame note (project summary sec.5): both PI's run_step_test
and LQI's run_step_test_lqi/_init_z0 already build Qbaseline/xhat0 from
the REAL equilibrium at whatever Q0_actual is passed in (via
gqc.initialize), not a hardcoded design-point value -- this is exactly
what that section warns needs to be right, so it is exercised for real
here for the first time (design-point tests never revealed a bug either
way, since design point == real equilibrium point there).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reduce_and_design_pi import AVR, Q0_OP, get_reduced_plant
from design_lqi import build_augmented, lqr_gain, kalman_gain, pick_observer_q_proc, _closed_loop_poles
import validate_pi as vpi
import validate_lqi as vlqi

KP_PI, KI_PI = 0.070, 0.600
QY_LQI, QI_LQI = 1.4, 20.0

Ar, Br, Cr, Dr, _ = get_reduced_plant(order=6, verbose=False)
Aaug, Baug = build_augmented(Ar, Br, Cr)
K_lqi, _, _ = lqr_gain(Aaug, Baug, Cr, QY_LQI, QI_LQI, eps=1e-3, r=1.0)
_ctrl_slowest = np.max(_closed_loop_poles(Aaug, Baug, K_lqi).real)
_q_proc = pick_observer_q_proc(Ar, Cr, Br, _ctrl_slowest * 4.0)
L_lqi = kalman_gain(Ar, Br, Cr, _q_proc, 1.0)


def _vref_trace_lqi(sol, Vref0, K):
    xhat = sol.y[11:17, :]; xi = sol.y[17, :]
    Kx, Ki = K[0, :6], K[0, 6]
    u_pi = -(Kx @ xhat) - Ki * xi
    return np.minimum(Vref0 + u_pi, AVR.VREF_MAX)


def _vref_trace_pi(sol, Vref0, Kp, Ki, qref_func):
    t = sol.t
    Qdm = sol.y[10, :]
    xi = sol.y[11, :]
    error = np.array([qref_func(tt) for tt in t]) - Qdm
    u_pi = Kp * error + Ki * xi
    return np.minimum(Vref0 + u_pi, AVR.VREF_MAX)


# =======================================================================
# ITEM 2: off-design setpoint steps
# =======================================================================
print("=" * 70)
print("ITEM 2: escalon de consigna fuera del punto de diseno (+0.15 pu)")
print("=" * 70)
offdesign_results = []
for Q0_test in [0.0, 0.6]:
    sol_pi, Qbase_pi, Vref0_pi = vpi.run_step_test(KP_PI, KI_PI, Q0_test, 0.15, t_total=6.0, t_event=1.0)
    ts_pi, ov_pi = vpi._step_settle_overshoot(sol_pi, Qbase_pi, 0.15)
    qref_pi = lambda t: Qbase_pi + (0.15 if t >= 1.0 else 0.0)
    vref_pi = _vref_trace_pi(sol_pi, Vref0_pi, KP_PI, KI_PI, qref_pi)

    sol_lqi, Qbase_lqi, Vref0_lqi = vlqi.run_step_test_lqi(Ar, Br, Cr, K_lqi, L_lqi, Q0_test, 0.15,
                                                            t_total=6.0, t_event=1.0)
    ts_lqi, ov_lqi = vlqi._metrics(sol_lqi, Qbase_lqi, 0.15)
    vref_lqi = _vref_trace_lqi(sol_lqi, Vref0_lqi, K_lqi)

    print(f"\nQ0={Q0_test:+.2f} (diseno fue en Q0={Q0_OP}):")
    print(f"  PI : t_settle={ts_pi:.3f}s  overshoot={ov_pi*100:6.2f}%  Vref max={np.max(vref_pi):.4f}")
    print(f"  LQI: t_settle={ts_lqi:.3f}s  overshoot={ov_lqi*100:6.2f}%  Vref max={np.max(vref_lqi):.4f}")
    offdesign_results.append((Q0_test, ts_pi, ov_pi, np.max(vref_pi), ts_lqi, ov_lqi, np.max(vref_lqi),
                               sol_pi, sol_lqi, Qbase_pi, Qbase_lqi, vref_pi, vref_lqi))

# =======================================================================
# ITEM 4: grid voltage (Einf) disturbance, Q_ref fixed
# =======================================================================
print("\n" + "=" * 70)
print("ITEM 4: perturbacion de tension de red (Einf), Q_ref fija")
print("=" * 70)
grid_results = []
for frac in [-0.10, -0.05, 0.05, 0.10]:
    sol_pi, Qbase_pi, Vref0_pi = vpi.run_grid_voltage_test(KP_PI, KI_PI, Q0_OP, frac, t_total=6.0, t_event=1.0)
    Qdm_pi = sol_pi.y[10, :]
    dev_pi = np.max(np.abs(Qdm_pi - Qbase_pi))
    final_err_pi = abs(Qdm_pi[-1] - Qbase_pi)
    qref_pi = lambda t: Qbase_pi
    vref_pi = _vref_trace_pi(sol_pi, Vref0_pi, KP_PI, KI_PI, qref_pi)

    sol_lqi, Qbase_lqi, Vref0_lqi = vlqi.run_grid_voltage_test_lqi(Ar, Br, Cr, K_lqi, L_lqi, Q0_OP,
                                                                    frac, t_total=6.0, t_event=1.0)
    Qdm_lqi = sol_lqi.y[10, :]
    dev_lqi = np.max(np.abs(Qdm_lqi - Qbase_lqi))
    final_err_lqi = abs(Qdm_lqi[-1] - Qbase_lqi)
    vref_lqi = _vref_trace_lqi(sol_lqi, Vref0_lqi, K_lqi)

    print(f"\nEinf {frac*100:+.0f}%:")
    print(f"  PI : desviacion max Q={dev_pi:.4f}  error final={final_err_pi:.2e}  Vref max={np.max(vref_pi):.4f} min={np.min(vref_pi):.4f}")
    print(f"  LQI: desviacion max Q={dev_lqi:.4f}  error final={final_err_lqi:.2e}  Vref max={np.max(vref_lqi):.4f} min={np.min(vref_lqi):.4f}")
    grid_results.append((frac, dev_pi, final_err_pi, dev_lqi, final_err_lqi))

# ---- plots ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for Q0_test, ts_pi, ov_pi, vmax_pi, ts_lqi, ov_lqi, vmax_lqi, sol_pi, sol_lqi, Qbase_pi, Qbase_lqi, _, _ in offdesign_results:
    axes[0, 0].plot(sol_pi.t, sol_pi.y[10, :], label=f'PI Q0={Q0_test}')
    axes[0, 0].plot(sol_lqi.t, sol_lqi.y[10, :], '--', label=f'LQI Q0={Q0_test}')
axes[0, 0].set_title('Item 2: Q_deliv_m, escalon fuera de diseno'); axes[0, 0].legend(fontsize=7)
axes[0, 0].set_xlabel('t [s]')

axes[0, 1].bar([f'PI\nQ0={r[0]}' for r in offdesign_results], [r[1] for r in offdesign_results],
               width=0.35, label='PI t_settle')
axes[0, 1].bar([f'LQI\nQ0={r[0]}' for r in offdesign_results], [r[4] for r in offdesign_results],
               width=0.35, label='LQI t_settle')
axes[0, 1].set_title('Item 2: tiempo de establecimiento'); axes[0, 1].legend(fontsize=7)

for frac, dev_pi, ferr_pi, dev_lqi, ferr_lqi in grid_results:
    pass
fracs = [r[0]*100 for r in grid_results]
axes[1, 0].plot(fracs, [r[1] for r in grid_results], 'o-', label='PI, desv. max Q')
axes[1, 0].plot(fracs, [r[3] for r in grid_results], 's-', label='LQI, desv. max Q')
axes[1, 0].set_title('Item 4: desviacion max de Q vs escalon en Einf'); axes[1, 0].legend(fontsize=8)
axes[1, 0].set_xlabel('escalon Einf [%]')

axes[1, 1].plot(fracs, [r[2] for r in grid_results], 'o-', label='PI, error final')
axes[1, 1].plot(fracs, [r[4] for r in grid_results], 's-', label='LQI, error final')
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('Item 4: error final de Q (log)'); axes[1, 1].legend(fontsize=8)
axes[1, 1].set_xlabel('escalon Einf [%]')

plt.tight_layout()
plt.savefig(_os.path.join(OUTPUTS_DIR, 'battery_offdesign_and_grid.png'), dpi=110)
print(f"\nGuardado outputs/battery_offdesign_and_grid.png")
