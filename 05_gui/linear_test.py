"""
linear_test.py
=================
Prueba de escalon RAPIDA sobre el modelo reducido (lineal), pensada
para el ciclo interactivo de reajuste de reguladores del paso 4 --NO
usa la planta no lineal completa (eso es el paso 5, mas lento y
exhaustivo). Discretizacion EXACTA (exponencial de matriz) tanto para
el observador como para la propia planta reducida, ya que todo es
lineal aqui -- rapido y sin aproximacion.
"""
import numpy as np
from scipy.linalg import expm
import control


def compute_lqi_gain(Ar, Br, Cr, qy, qi):
    """Un unico calculo de K (LQR) para un qy,qi concreto -- sin barrido."""
    n = Ar.shape[0]
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = Ar
    A_aug[n, :n] = -Cr[0, :]
    B_aug = np.vstack([Br, [[0.0]]])
    Cy = np.zeros((1, n + 1))
    Cy[0, :n] = Cr[0, :]
    Qx = qy * (Cy.T @ Cy)
    Qx[n, n] += qi
    Qx = 0.5 * (Qx + Qx.T)
    K, S, E = control.lqr(A_aug, B_aug, Qx, np.array([[1.0]]))
    poles = np.linalg.eigvals(A_aug - B_aug @ K)
    return K, poles


def compute_lqi_observer(Ar, Cr, qn_scale=1.0, rn=1e-4):
    n = Ar.shape[0]
    L, _, _ = control.lqe(Ar, np.eye(n), Cr, qn_scale * np.eye(n), np.array([[rn]]))
    return L


def _discretize_linear(A, b, Ts):
    """[x(Ts); 1] = expm([[A,b],[0,0]]*Ts) @ [x0;1] -- exacto, para
    dx/dt = A@x + b (b constante durante el paso)."""
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    return expm(M * Ts)


def linear_step_test(controller_type, Ar, Br, Cr, Ts, t_total, step_size, t_event,
                      Vref0, VREF_MIN, VREF_MAX, K=None, L=None, Kp=None, Ki=None):
    """Devuelve dict con t, Q (desviacion + Qeq=0 implicito), Vref.
    Todo en variables de desviacion (el punto de equilibrio es Q=0,
    Vref=Vref0) -- coherente con como se disenan los reguladores."""
    n = Ar.shape[0]
    n_steps = int(np.round(t_total / Ts))
    log = {'t': [], 'Q': [], 'Vref': []}

    x_plant = np.zeros(n)
    Vref = Vref0
    xi = 0.0
    x_hat = np.zeros(n) if controller_type == 'LQI' else None

    if controller_type == 'LQI':
        Kx, Ki_ = K[0, :n], K[0, n]
        Aobs = Ar - L.flatten()[:, None] @ Cr

    def record(t, x_plant_now, Vref_now):
        y = float(np.atleast_1d(Cr @ x_plant_now).item())
        log['t'].append(t); log['Q'].append(y); log['Vref'].append(Vref_now)

    record(0.0, x_plant, Vref)
    for k in range(n_steps):
        t_start, t_end = k * Ts, (k + 1) * Ts
        y_meas = float(np.atleast_1d(Cr @ x_plant).item())
        Qref_now = step_size if t_start >= t_event else 0.0

        if controller_type == 'PI':
            error = Qref_now - y_meas
            xi += error * Ts
            Vref_unclamped = Vref0 + Kp * error + Ki * xi
            Vref = float(np.clip(Vref_unclamped, VREF_MIN, VREF_MAX))
            if Ki != 0.0:
                xi -= (Vref - Vref_unclamped) / Ki

        elif controller_type == 'LQI':
            u_prev = Vref - Vref0
            b_obs = Br.flatten() * u_prev + L.flatten() * y_meas
            expM_obs = _discretize_linear(Aobs, b_obs, Ts)
            x_hat = expM_obs[:n, :n] @ x_hat + expM_obs[:n, n]
            xi += (Qref_now - y_meas) * Ts
            Vref_unclamped = Vref0 - float(Kx @ x_hat) - Ki_ * xi
            Vref = float(np.clip(Vref_unclamped, VREF_MIN, VREF_MAX))
            if Ki_ != 0.0:
                xi -= (Vref - Vref_unclamped) / Ki_
        else:
            raise ValueError(controller_type)

        b_plant = Br.flatten() * (Vref - Vref0)
        expM_plant = _discretize_linear(Ar, b_plant, Ts)
        x_plant = expM_plant[:n, :n] @ x_plant + expM_plant[:n, n]
        record(t_end, x_plant, Vref)

    return {k: np.array(v) for k, v in log.items()}
