import sys as _sys, os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ['01_model', '02_linearization']:
    _p = _os.path.join(_PKG_ROOT, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
"""
verify_linearize_condenser.py
================================
The critical test: compare the symbolic A (11x11)/B (11x1)/C (5x11)
matrices from linearize_condenser.py against a finite-difference
linearization of the ACTUAL closed_loop_derivatives() function -- the
exact nonlinear RHS, built directly from genqec_model.py + controls.py,
no symbolic machinery involved -- at the SAME operating point.

Machine parameters here are the SAME illustrative placeholder set used
in linearize_condenser.py's own __main__, EXCEPT H=15/D=0 which ARE
real (per chat). AVR and load parameters are the real, given ones.
"""
import numpy as np
import genqec_model as gqc
import controls as ctrl
import linearize_condenser as lc

sat = gqc.make_saturation('quadratic', 0.10, 0.30)
p = gqc.GENQECParams(Ra=0.003, Xl=0.15, Xd=1.80, Xdp=0.30, Xdpp=0.22,
                      Xq=1.70, Xqp=0.50, Xqpp=0.25,
                      Tdop=7.5, Tdopp=0.03, Tqop=0.50, Tqopp=0.05,
                      H=15.0, D=0.0, Kw=0.20, sat=sat, f0=60.0)
AVR = ctrl.AVRParams()   # real: Kp=72, Ki=35, U=[-2.6,3.25], KC=0.0308, FACTOR_MAX=1.4, T_MEAS=0.0111, VREF_MAX=1.15

Rt, Xt, Rline, Xline, Rgrid, Xgrid = 0.002, 0.12, 0.01, 0.08, 0.01, 0.06
Re_total, Xe_total = Rt + Rline + Rgrid, Xt + Xline + Xgrid
R1, X1 = Rt + Rline, Xt + Xline
Vt0, P0, Q0 = 1.0, 0.0, 0.35

# local load: 8% of condenser nominal, cos(phi)=0.85 lagging, at V=1.0pu
S_load, cosphi = 0.08, 0.85
P_load, Q_load = S_load * cosphi, S_load * (1 - cosphi ** 2) ** 0.5
Gload, Bload = P_load / 1.0 ** 2, -Q_load / 1.0 ** 2
Tdeliv_val = 0.08

# ---- symbolic linearization ----
result = lc.linearize_at_operating_point(p, AVR, Re_total, Xe_total, R1, X1,
                                          Vt0, P0, Q0, 0.10, 0.30, 'quadratic',
                                          Gload=Gload, Bload=Bload, Tdeliv_val=Tdeliv_val)
A_sym, B_sym, C_sym = result['A'], result['B'], result['C']
op = result['operating_point']

# ---- build the SAME operating point as a state vector (11 states) ----
state0 = np.array([op['delta0'], op['omega0'], op['Eqp0'], op['psidp0'], op['Edp0'], op['psiqp0']])
Efd0, Pmech0, Einf_val, Vref0 = op['Efd0'], op['Pmech0'], op['Einf'], op['Vref0']

avr_x0, Vref0_check = ctrl.avr_initialize(op['Vtgen0'], Efd0, op['Ifd0'], AVR)
assert abs(Vref0_check - Vref0) < 1e-9
deliv_x0 = np.array([op['Vdeliv_raw0'], op['Qdeliv_raw0']])
x_full0 = np.concatenate([state0, avr_x0, deliv_x0])
n = len(x_full0)
print(f"n={n} states, |x0|={np.linalg.norm(x_full0):.4f}")


def outputs_raw(Vd, Vq, Id, Iq):
    """Load-corrected delivery-bus outputs + raw Vt_generator (Part 6)."""
    Id_load = Gload * Vd - Bload * Vq
    Iq_load = Gload * Vq + Bload * Vd
    Id_net, Iq_net = Id - Id_load, Iq - Iq_load
    Vd_deliv = Vd - R1 * Id_net + X1 * Iq_net
    Vq_deliv = Vq - R1 * Iq_net - X1 * Id_net
    V_deliv = np.hypot(Vd_deliv, Vq_deliv)
    Q_deliv = Vq_deliv * Id_net - Vd_deliv * Iq_net
    Vt_gen = np.hypot(Vd, Vq)
    return V_deliv, Q_deliv, Vt_gen


def closed_loop_derivatives(t, x_full, p, avr_p, Pmech, Einf, Re_total, Xe_total, vref_func):
    x6 = x_full[:6]
    x_avr = x_full[6:9]
    x_deliv = x_full[9:11]
    delta, omega, Eqp, psidp, Edp, psiqp = x6
    Id, Iq, Vdterm, Vqterm, Sa, psi_ag = gqc.solve_network(
        delta, omega, Eqp, psidp, Edp, psiqp, p, Einf, Re_total, Xe_total, Gf=Gload, Bf=Bload)
    Vtgen = float(np.hypot(Vdterm, Vqterm))
    Ifd = gqc.field_current(Eqp, psidp, Id, Sa, p)
    Vref_now = vref_func(t)
    dx_avr, Efd = ctrl.avr_derivatives(x_avr, Vref_now, Vtgen, Ifd, avr_p)
    dx6 = gqc.derivatives(t, x6, p, Efd, Pmech, Einf, Re_total, Xe_total, Gf=Gload, Bf=Bload)

    V_deliv_raw, Q_deliv_raw, _ = outputs_raw(Vdterm, Vqterm, Id, Iq)
    dVdeliv_m = (V_deliv_raw - x_deliv[0]) / Tdeliv_val
    dQdeliv_m = (Q_deliv_raw - x_deliv[1]) / Tdeliv_val

    return np.concatenate([dx6, dx_avr, [dVdeliv_m, dQdeliv_m]])


def rhs(x, vref):
    return closed_loop_derivatives(0.0, x, p, AVR, Pmech0, Einf_val,
                                    Re_total, Xe_total, lambda t: vref)


f0 = rhs(x_full0, Vref0)
print(f"Equilibrium residual (should be ~0): {np.max(np.abs(f0)):.2e}\n")

# ---- finite-difference Jacobian (central differences) ----
h = 1e-6
A_fd = np.zeros((n, n))
for j in range(n):
    xp = x_full0.copy(); xp[j] += h
    xm = x_full0.copy(); xm[j] -= h
    A_fd[:, j] = (rhs(xp, Vref0) - rhs(xm, Vref0)) / (2 * h)

hB = 1e-6
B_fd = ((rhs(x_full0, Vref0 + hB) - rhs(x_full0, Vref0 - hB)) / (2 * hB)).reshape(-1, 1)

print("=" * 90)
print(f"{'row':4s}{'col':4s}{'state':10s}{'wrt':10s}{'symbolic':>14s}{'finite-diff':>14s}{'abs err':>12s}")
print("=" * 90)
max_abs_err = 0.0
worst = None
names = result['state_names']
for i in range(n):
    for j in range(n):
        sym_v, fd_v = A_sym[i, j], A_fd[i, j]
        err = abs(sym_v - fd_v)
        scale = max(abs(fd_v), 1.0)
        if err / scale > 1e-3:
            print(f"{i:<4d}{j:<4d}{names[i]:10s}{names[j]:10s}{sym_v:14.5f}{fd_v:14.5f}{err:12.2e}  <<<")
        if err > max_abs_err:
            max_abs_err = err
            worst = (i, j)
print(f"\nMax |A_sym - A_fd| over all {n*n} entries: {max_abs_err:.3e}  at {worst} "
      f"({names[worst[0]]}/{names[worst[1]]})  sym={A_sym[worst[0],worst[1]]:.5f} fd={A_fd[worst[0],worst[1]]:.5f}")

print("\nB column (dx/dVref): symbolic vs finite-difference")
maxBerr = 0.0
for i in range(n):
    err = abs(B_sym[i, 0] - B_fd[i, 0])
    maxBerr = max(maxBerr, err)
    flag = "  <<<" if err > 1e-3 * max(abs(B_fd[i, 0]), 1.0) else ""
    print(f"  {names[i]:10s} sym={B_sym[i,0]:12.5f}  fd={B_fd[i,0]:12.5f}  err={err:.2e}{flag}")
print(f"\nMax |B_sym - B_fd|: {maxBerr:.3e}")


# ---- output matrix C: finite-difference the 5 outputs directly ----
def outputs(x_full):
    x6 = x_full[:6]
    x_deliv = x_full[9:11]
    Id, Iq, Vd, Vq, Sa, _ = gqc.solve_network(*x6, p, Einf_val, Re_total, Xe_total, Gf=Gload, Bf=Bload)
    V_deliv, Q_deliv, Vt_gen = outputs_raw(Vd, Vq, Id, Iq)
    return np.array([V_deliv, Q_deliv, Vt_gen, x_deliv[0], x_deliv[1]])


C_fd = np.zeros((5, n))
for j in range(n):
    xp = x_full0.copy(); xp[j] += h
    xm = x_full0.copy(); xm[j] -= h
    C_fd[:, j] = (outputs(xp) - outputs(xm)) / (2 * h)

print("\nC matrix: symbolic vs finite-difference")
maxCerr = 0.0
for i, oname in enumerate(result['output_names']):
    for j in range(n):
        err = abs(C_sym[i, j] - C_fd[i, j])
        maxCerr = max(maxCerr, err)
        if err > 1e-3 * max(abs(C_fd[i, j]), 1.0):
            print(f"  {oname:14s} wrt {names[j]:10s} sym={C_sym[i,j]:12.5f}  fd={C_fd[i,j]:12.5f}  err={err:.2e}  <<<")
print(f"\nMax |C_sym - C_fd|: {maxCerr:.3e}")

print("\n" + "=" * 90)
print("OVERALL:", "PASS" if max(max_abs_err, maxBerr, maxCerr) < 1e-2 else "FAIL -- see MISMATCH rows above")
