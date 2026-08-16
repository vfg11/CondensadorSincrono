"""
plc_battery.py
=================
Simulacion de lazo cerrado FIEL A COMO SE EJECUTARIA EN UN PLC: el
regulador (PI, LQI o Rele) se muestrea y actualiza UNA VEZ POR CICLO
(Ts, 100ms por defecto), y Vref se mantiene FIJO durante todo el ciclo
mientras la planta (maquina+AVR+retardos de medida) se integra en
continuo dentro de ese intervalo. Esto es DISTINTO de como simulan
validate_pi.py/validate_lqi.py/validate_relay.py del paquete original,
donde el regulador se recalcula continuamente dentro de la propia
solve_ivp -- valido para ver el comportamiento ideal del diseno, pero
no representa lo que un PLC real hace.

La FISICA de la planta (extraida de closed_loop_derivatives de
validate_pi.py, sin cambios) es identica para los tres reguladores; lo
unico que cambia es que Vref, en vez de calcularse dentro de la propia
EDO, se calcula UNA VEZ al principio de cada ciclo Ts y se pasa como
parametro fijo.
"""
import numpy as np
from scipy.linalg import expm as sla_expm
from scipy.integrate import solve_ivp

import genqec_model as gqc
import controls as ctrl


def plant_rhs(t, x_full, Pmech0, Einf_val, Gf, Bf, Vref_fixed, P, AVR, Re_total, Xe_total, TDELIV):
    """11 estados: maquina(6) + avr(3, Vt_m/Ifd_m/x_I interno) + retardos
    de entrega (2, Vdeliv_m/Qdeliv_m). Vref_fixed es un PARAMETRO, no un
    estado -- se mantiene constante durante todo el ciclo Ts en el que
    se llama esta funcion."""
    x6 = x_full[:6]
    x_avr = x_full[6:9]
    x_deliv = x_full[9:11]

    delta, omega, Eqp, psidp, Edp, psiqp = x6
    Id, Iq, Vd, Vq, Sa, _ = gqc.solve_network(delta, omega, Eqp, psidp, Edp, psiqp,
                                                P, Einf_val, Re_total, Xe_total, Gf=Gf, Bf=Bf)
    Vtgen = float(np.hypot(Vd, Vq))
    Ifd = gqc.field_current(Eqp, psidp, Id, Sa, P)

    dx_avr, Efd = ctrl.avr_derivatives(x_avr, Vref_fixed, Vtgen, Ifd, AVR)
    dx6 = gqc.derivatives(t, x6, P, Efd, Pmech0, Einf_val, Re_total, Xe_total, Gf=Gf, Bf=Bf)

    Id_load = Gf * Vd - Bf * Vq
    Iq_load = Gf * Vq + Bf * Vd
    Id_net, Iq_net = Id - Id_load, Iq - Iq_load
    R1 = getattr(plant_rhs, '_R1', None)
    X1 = getattr(plant_rhs, '_X1', None)
    Vd_deliv = Vd - R1 * Id_net + X1 * Iq_net
    Vq_deliv = Vq - R1 * Iq_net - X1 * Id_net
    V_deliv_raw = float(np.hypot(Vd_deliv, Vq_deliv))
    Q_deliv_raw = float(Vq_deliv * Id_net - Vd_deliv * Iq_net)

    dVdeliv_m = (V_deliv_raw - x_deliv[0]) / TDELIV
    dQdeliv_m = (Q_deliv_raw - x_deliv[1]) / TDELIV

    return np.concatenate([dx6, dx_avr, [dVdeliv_m, dQdeliv_m]])


def measure(x_full):
    return dict(Vt_m=x_full[6], Ifd_m=x_full[7], Vdeliv_m=x_full[9], Qdeliv_m=x_full[10])


def init_plant_plc(params, Q0_actual):
    """params: objeto/dict con P (GENQECParams), AVR (AVRParams),
    Re_total, Xe_total, R1, X1, Vt0, P0_OP, GLOAD, BLOAD, TDELIV."""
    P, AVR = params['P'], params['AVR']
    Re_total, Xe_total = params['Re_total'], params['Xe_total']
    R1, X1 = params['R1'], params['X1']
    Vt0, P0_OP = params['Vt0'], params['P0_OP']
    GLOAD, BLOAD = params['GLOAD'], params['BLOAD']

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

    x_full0 = np.concatenate([state0, avr_x0, [Vdeliv_raw0, Qdeliv_raw0]])
    return x_full0, Pmech0, Einf_val, Vref0, Qdeliv_raw0


def _integrate_plant(x_full, t_start, t_end, params, Einf_val, Gf, Bf, Vref_fixed, Pmech0):
    plant_rhs._R1 = params['R1']
    plant_rhs._X1 = params['X1']
    sol = solve_ivp(plant_rhs, [t_start, t_end], x_full,
                     args=(Pmech0, Einf_val, Gf, Bf, Vref_fixed, params['P'], params['AVR'],
                           params['Re_total'], params['Xe_total'], params['TDELIV']),
                     method='Radau', max_step=(t_end - t_start) / 2, rtol=1e-7, atol=1e-9)
    return sol.y[:, -1]


def run_plc_test(controller_type, gains, params, Q0_actual, qref_func, Ts, t_total,
                  einf_schedule=None, fault_schedule=None):
    """
    Simulacion generica con muestreo real a Ts.

    controller_type: 'PI', 'LQI' o 'Rele'
    gains: dict especifico de cada tipo --
        PI:   {'Kp':.., 'Ki':..}
        LQI:  {'K':.., 'L':.., 'Ar':.., 'Br':.., 'Cr':.., 'Qeq':.., 'Vt0':..}
        Rele: {'dead_band':.., 'rate':..}
    qref_func(t): consigna de Q en cada instante (evaluada al INICIO de
        cada ciclo, como haria un PLC)
    einf_schedule(t): tension de red en cada instante (por defecto fija)
    fault_schedule(t): devuelve (Gf_extra, Bf_extra) para simular un
        cortocircuito en el instante t (por defecto 0,0)
    """
    AVR = params['AVR']
    x_full, Pmech0, Einf_base, Vref0, Qbaseline = init_plant_plc(params, Q0_actual)
    if einf_schedule is None:
        einf_schedule = lambda t: Einf_base
    if fault_schedule is None:
        fault_schedule = lambda t: (0.0, 0.0)

    n_steps = int(np.round(t_total / Ts))
    log = {k: [] for k in ['t', 'Q', 'Vref', 'Vt', 'Einf']}

    Vref = Vref0
    xi_pi = 0.0
    x_hat = np.zeros(gains['K'].shape[1] - 1) if controller_type == 'LQI' else None
    xi_lqi = 0.0

    def record(t, x_full_now, Vref_now, Einf_now):
        m = measure(x_full_now)
        log['t'].append(t); log['Q'].append(m['Qdeliv_m']); log['Vref'].append(Vref_now)
        log['Vt'].append(m['Vt_m']); log['Einf'].append(Einf_now)

    record(0.0, x_full, Vref, Einf_base)
    for k in range(n_steps):
        t_start, t_end = k * Ts, (k + 1) * Ts
        y = measure(x_full)  # muestra UNA vez al principio del ciclo -- como un PLC real
        Qref_now = qref_func(t_start)

        if controller_type == 'PI':
            Kp, Ki = gains['Kp'], gains['Ki']
            error = Qref_now - y['Qdeliv_m']
            xi_pi += error * Ts
            Vref_unclamped = Vref0 + Kp * error + Ki * xi_pi
            Vref = float(np.clip(Vref_unclamped, AVR.VREF_MIN, AVR.VREF_MAX))
            if Ki != 0.0:
                xi_pi -= (Vref - Vref_unclamped) / Ki

        elif controller_type == 'LQI':
            Ar, Br, Cr = gains['Ar'], gains['Br'], gains['Cr']
            K, L = gains['K'], gains['L']
            Qeq = gains['Qeq']
            u_prev = Vref - Vref0
            y_meas = y['Qdeliv_m'] - Qeq
            L_flat = L.flatten()
            # Discretizacion EXACTA del observador via exponencial de
            # matriz (Van Loan), no una aproximacion de Euler -- ver
            # chat: sub-pasos de Euler resultaron insuficientes incluso
            # con N=10 para las ganancias recien disenadas en orden 7
            # (polos mas rapidos que el caso ya validado). exp() es
            # incondicionalmente estable, exacto para cualquier Ts,
            # sea cual sea la velocidad de los polos.
            n_obs = Ar.shape[0]
            Aobs = Ar - np.outer(L_flat, Cr.flatten())  # dinamica del ERROR de estimacion
            Mblock = np.zeros((n_obs + 1, n_obs + 1))
            Mblock[:n_obs, :n_obs] = Aobs
            Mblock[:n_obs, n_obs] = Br.flatten() * u_prev + L_flat * y_meas
            expM = sla_expm(Mblock * Ts)
            x_hat = expM[:n_obs, :n_obs] @ x_hat + expM[:n_obs, n_obs]
            xi_lqi += (Qref_now - y['Qdeliv_m']) * Ts
            Kx, Ki_ = K[0, :-1], K[0, -1]
            Vref_unclamped = Vref0 - float(Kx @ x_hat) - Ki_ * xi_lqi
            Vref = float(np.clip(Vref_unclamped, AVR.VREF_MIN, AVR.VREF_MAX))
            if Ki_ != 0.0:
                xi_lqi -= (Vref - Vref_unclamped) / Ki_

        elif controller_type == 'Rele':
            db, rate = gains['dead_band'], gains['rate']
            error = Qref_now - y['Qdeliv_m']
            relay_out = 1.0 if error > db else (-1.0 if error < -db else 0.0)
            Vref_unclamped = Vref + rate * relay_out * Ts
            Vref = float(np.clip(Vref_unclamped, AVR.VREF_MIN, AVR.VREF_MAX))

        else:
            raise ValueError(f"controller_type desconocido: {controller_type}")

        Gf_extra, Bf_extra = fault_schedule(t_start)
        Einf_now = einf_schedule(t_start)
        x_full = _integrate_plant(x_full, t_start, t_end, params, Einf_now,
                                   params['GLOAD'] + Gf_extra, params['BLOAD'] + Bf_extra,
                                   Vref, Pmech0)
        record(t_end, x_full, Vref, Einf_now)

    return {k: np.array(v) for k, v in log.items()}, Qbaseline, Vref0
