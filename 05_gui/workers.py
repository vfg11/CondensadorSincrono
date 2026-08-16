"""
workers.py
============
Workers en QThread para los pasos largos: linealizacion, reduccion (con
sugerencia de orden), y diseno de reguladores. Cada uno emite senales
de progreso/resultado/error para no bloquear la ventana principal.
"""
import sys, os
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PKG_ROOT = os.path.dirname(_HERE)
    for _d in ['01_model', '02_linearization', '03_design']:
        _p = os.path.join(_PKG_ROOT, _d)
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
except Exception:
    pass

import numpy as np
import scipy.linalg as sla
from PySide6.QtCore import QThread, Signal

from linearize_condenser import linearize_at_operating_point
import control


class WorkerThread(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, fn, kwargs):
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(self.progress.emit, **self.kwargs)
            self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


def do_linearize(report, objs):
    report("Derivando y linealizando (puede tardar ~10s)...")
    r = linearize_at_operating_point(
        objs['P'], objs['AVR'], objs['Re_total'], objs['Xe_total'], objs['R1'], objs['X1'],
        objs['Vt0'], objs['P0_OP'], objs['Q0_OP'], 0.10, 0.30, 'quadratic',
        Gload=objs['GLOAD'], Bload=objs['BLOAD'], Tdeliv_val=objs['TDELIV'])
    report("Linealizacion completa.")
    return dict(A=r['A'], B=r['B'], C=r['C'], D=r['D'],
                state_names=r['state_names'], output_names=r['output_names'],
                flags=r['flags'], Vref0=r['operating_point']['Vref0'])


def compute_hankel_spectrum(A, B, C_row):
    def chol_psd(M):
        w, v = np.linalg.eigh(M)
        w = np.clip(w, 0, None)
        return v @ np.diag(np.sqrt(w))
    Wc = sla.solve_lyapunov(A, -B @ B.T)
    Wo = sla.solve_lyapunov(A.T, -C_row.T @ C_row)
    Lc, Lo = chol_psd(Wc), chol_psd(Wo)
    U, S, Vt = np.linalg.svd(Lc.T @ Lo)
    return S


def suggest_order(hsv, max_order=None, noise_floor_rel=1e-8):
    """Sugiere el orden con el mayor salto RELATIVO entre valores
    singulares de Hankel consecutivos (el 'codo' natural del espectro).
    Ignora huecos calculados con valores por debajo del suelo de ruido
    numerico (relativo al mayor valor singular) -- sin esto, ratios
    entre valores ya-practicamente-cero dan saltos espurios enormes que
    no reflejan ningun 'codo' real del espectro."""
    n = len(hsv)
    if max_order is None:
        max_order = n - 1
    floor = hsv[0] * noise_floor_rel
    valid = hsv > floor
    ratios = np.where(valid[1:], hsv[:-1] / np.maximum(hsv[1:], 1e-300), 0.0)
    ratios = ratios[:max_order]
    if not np.any(ratios > 0):
        return min(3, max_order)  # fallback razonable si todo es ruido
    best_idx = int(np.argmax(ratios))
    return best_idx + 1


def do_reduce(report, A, B, C_row, order):
    report("Calculando espectro de Hankel...")
    hsv = compute_hankel_spectrum(A, B, C_row)
    suggested = suggest_order(hsv)
    order_to_use = order if order is not None else suggested
    report(f"Reduciendo a orden {order_to_use} (sugerido: {suggested})...")
    sys_full = control.ss(A, B, C_row, np.zeros((1, 1)))
    sys_r = control.balred(sys_full, order_to_use, method='matchdc')
    Ar, Br, Cr, Dr = sys_r.A, sys_r.B, sys_r.C, sys_r.D
    dc_full = float(control.dcgain(sys_full))
    dc_red = float(control.dcgain(sys_r))
    report("Reduccion completa.")
    return dict(hsv=hsv, suggested_order=suggested, order_used=order_to_use,
                Ar=Ar, Br=Br, Cr=Cr, Dr=Dr, dc_full=dc_full, dc_red=dc_red)


def do_design(report, Ar, Br, Cr, Ts, qy_grid=None, qi_grid=None, kp_grid=None, ki_grid=None):
    report("Disenando LQI (LQR + Kalman)...")
    n = Ar.shape[0]
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = Ar
    A_aug[n, :n] = -Cr[0, :]
    B_aug = np.vstack([Br, [[0.0]]])
    Cy = np.zeros((1, n + 1))
    Cy[0, :n] = Cr[0, :]

    if qy_grid is None:
        qy_grid = [0.5, 1.0, 1.4, 2.0, 3.0]
    if qi_grid is None:
        qi_grid = [5.0, 10.0, 20.0, 50.0, 100.0]

    best = None
    for qy in qy_grid:
        for qi in qi_grid:
            Qx = qy * (Cy.T @ Cy)
            Qx[n, n] += qi
            Qx = 0.5 * (Qx + Qx.T)
            try:
                K, S, E = control.lqr(A_aug, B_aug, Qx, np.array([[1.0]]))
            except Exception:
                continue
            poles = np.linalg.eigvals(A_aug - B_aug @ K)
            if np.max(poles.real) >= -1e-6:
                continue
            settle = -1.0 / np.max(poles.real)
            if best is None or settle < best[0]:
                best = (settle, qy, qi, K)
    if best is None:
        raise RuntimeError("No se encontro ninguna combinacion qy/qi estable en la rejilla.")
    _, qy_best, qi_best, K = best

    QN = np.eye(n)
    RN = np.array([[1e-4]])
    L, _, _ = control.lqe(Ar, np.eye(n), Cr, QN, RN)
    report(f"LQI: qy={qy_best}, qi={qi_best}")

    report("Disenando PI (barrido en rejilla sobre el modelo reducido)...")
    if kp_grid is None:
        kp_grid = [0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25]
    if ki_grid is None:
        ki_grid = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2]
    sys_r = control.ss(Ar, Br, Cr, np.zeros((1, 1)))
    t_sweep = np.linspace(0, 30, 3000)
    best_pi = None
    for kp in kp_grid:
        for ki in ki_grid:
            C_pi = control.tf([kp, ki], [1, 0])
            try:
                Tcl = control.feedback(control.series(C_pi, sys_r), 1)
            except Exception:
                continue
            if not np.all(np.linalg.eigvals(Tcl.A).real < -1e-9):
                continue
            try:
                _, y = control.step_response(Tcl, T=t_sweep)
            except Exception:
                continue
            if abs(y[-1] - 1.0) > 0.02:
                continue
            settled = np.where(np.abs(y - y[-1]) > 0.02)[0]
            ts = t_sweep[settled[-1]] if len(settled) else 0.0
            if best_pi is None or ts < best_pi[0]:
                best_pi = (ts, kp, ki)
    if best_pi is None:
        raise RuntimeError("No se encontro ninguna combinacion Kp/Ki estable en la rejilla.")
    _, kp_best, ki_best = best_pi
    report(f"PI: Kp={kp_best}, Ki={ki_best}")

    return dict(K=K, L=L, qy=qy_best, qi=qi_best, Kp=kp_best, Ki=ki_best, Ts=Ts)
