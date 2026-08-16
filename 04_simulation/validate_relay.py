import sys as _sys, os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ['01_model', '02_linearization', '03_design', '04_simulation']:
    _p = _os.path.join(_PKG_ROOT, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
OUTPUTS_DIR = _os.path.join(_PKG_ROOT, 'outputs')
_os.makedirs(OUTPUTS_DIR, exist_ok=True)
"""
validate_relay.py
====================
Tercer regulador: rele de tres posiciones (subir/mantener/bajar) con
banda muerta:
    error = Qref - Qdeliv_m
    salida_rele = +1 si error > +DEAD_BAND, -1 si error < -DEAD_BAND, 0 si no
    dVref/dt = RATE * salida_rele
    Vref = integral, saturada en [VREF_MIN, VREF_MAX]
"""
import numpy as np
from scipy.integrate import solve_ivp

import genqec_model as gqc
import controls as ctrl
from reduce_and_design_pi import (P_REAL as P, AVR, Re_total, Xe_total, R1, X1,
                                   Vt0, P0_OP, Q0_OP, GLOAD, BLOAD, TDELIV)
from validate_pi import init_plant, outputs_raw

DEAD_BAND = 0.01
RATE = 0.002


def closed_loop_derivatives_relay(t, x_full, Pmech0, Einf_t, qref_func, Gf_t, Bf_t,
                                   dead_band=DEAD_BAND, rate=RATE):
    x6 = x_full[:6]
    x_avr = x_full[6:9]
    x_deliv = x_full[9:11]
    Vref = float(np.clip(x_full[11], AVR.VREF_MIN, AVR.VREF_MAX))

    Gf, Bf = Gf_t(t), Bf_t(t)
    Einf_val = Einf_t(t)
    delta, omega, Eqp, psidp, Edp, psiqp = x6
    Id, Iq, Vd, Vq, Sa, _ = gqc.solve_network(delta, omega, Eqp, psidp, Edp, psiqp,
                                                P, Einf_val, Re_total, Xe_total, Gf=Gf, Bf=Bf)
    Vtgen = float(np.hypot(Vd, Vq))
    Ifd = gqc.field_current(Eqp, psidp, Id, Sa, P)

    Qref = qref_func(t)
    error = Qref - x_deliv[1]

    if error > dead_band:
        relay_out = 1.0
    elif error < -dead_band:
        relay_out = -1.0
    else:
        relay_out = 0.0

    dVref_raw = rate * relay_out
    at_max = x_full[11] >= AVR.VREF_MAX and dVref_raw > 0
    at_min = x_full[11] <= AVR.VREF_MIN and dVref_raw < 0
    dVref = 0.0 if (at_max or at_min) else dVref_raw

    dx_avr, Efd = ctrl.avr_derivatives(x_avr, Vref, Vtgen, Ifd, AVR)
    dx6 = gqc.derivatives(t, x6, P, Efd, Pmech0, Einf_val, Re_total, Xe_total, Gf=Gf, Bf=Bf)

    V_deliv_raw, Q_deliv_raw = outputs_raw(Vd, Vq, Id, Iq)
    dVdeliv_m = (V_deliv_raw - x_deliv[0]) / TDELIV
    dQdeliv_m = (Q_deliv_raw - x_deliv[1]) / TDELIV

    return np.concatenate([dx6, dx_avr, [dVdeliv_m, dQdeliv_m], [dVref]])


def run_step_test_relay(Q0_actual, step_size, t_total=10.0, t_event=1.0):
    x_full0, Efd0, Pmech0, Einf_val, Vref0, Qbaseline = init_plant(Q0_actual)
    x_full0[11] = Vref0
    qref_func = lambda t: Qbaseline + (step_size if t >= t_event else 0.0)
    Gf_t, Bf_t = (lambda t: GLOAD), (lambda t: BLOAD)
    Einf_t = lambda t: Einf_val
    sol = solve_ivp(closed_loop_derivatives_relay, [0, t_total], x_full0, method='Radau',
                     args=(Pmech0, Einf_t, qref_func, Gf_t, Bf_t),
                     max_step=0.02, rtol=1e-8, atol=1e-10, dense_output=True)
    return sol, Qbaseline, Vref0


def run_fault_test_relay(Q0_actual, fault_duration=0.15, Gfault=25.0, t_total=12.0, t_fault=1.0):
    x_full0, Efd0, Pmech0, Einf_val, Vref0, Qbaseline = init_plant(Q0_actual)
    x_full0[11] = Vref0
    qref_func = lambda t: Qbaseline
    Gf_t = lambda t: GLOAD + (Gfault if t_fault <= t < t_fault + fault_duration else 0.0)
    Bf_t = lambda t: BLOAD
    Einf_t = lambda t: Einf_val
    sol = solve_ivp(closed_loop_derivatives_relay, [0, t_total], x_full0, method='Radau',
                     args=(Pmech0, Einf_t, qref_func, Gf_t, Bf_t),
                     max_step=0.02, rtol=1e-8, atol=1e-10, dense_output=True)
    return sol, Qbaseline, Vref0
