import sys as _sys, os as _os
try:
    _PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    for _d in ['01_model', '02_linearization', '03_design', '04_simulation', '05_gui']:
        _p = _os.path.join(_PKG_ROOT, _d)
        if _os.path.isdir(_p) and _p not in _sys.path:
            _sys.path.insert(0, _p)
except Exception:
    _PKG_ROOT = None

# Detecta si esto corre como ejecutable compilado (Nuitka define
# __compiled__; PyInstaller define sys.frozen) para decidir donde
# escribir OUTPUTS_DIR -- junto al ejecutable en ese caso, junto al
# codigo fuente en ejecucion normal.
_is_compiled = '__compiled__' in globals() or getattr(_sys, 'frozen', False)
if _is_compiled or _PKG_ROOT is None:
    OUTPUTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(_sys.argv[0])), 'outputs')
else:
    OUTPUTS_DIR = _os.path.join(_PKG_ROOT, 'outputs')
_os.makedirs(OUTPUTS_DIR, exist_ok=True)
"""
linearize_condenser.py
=========================
Full symbolic linearization of the closed-loop synchronous-condenser
system: GENQEC (6 states, UNCHANGED from the previous project -- same
machine model structure, only numeric parameters differ per machine)
+ new PI-based AVR (3 states) = 9 states, single input Vref, outputs
[Vdelivery, Q_delivered, Vt_generator]. PSS is out of scope for this
project (disabled) -- not implemented at all, not just zeroed.

Method (unchanged from the previous project): every algebraic quantity
(Id,Iq,Vd,Vq,Sa,Ifd,Efd) is differentiated using STAGED chain rule
through abstract intermediate symbols -- never asking sympy to simplify
through a fully-nested expression at once, which was found to make
Matrix.solve()/diff() effectively hang. The network+saturation
algebraic loop (Id,Iq,Vd,Vq,Sa) is handled with the implicit function
theorem, exactly as in the previous project (PART 1 below is reused
byte-for-byte -- it is machine-model physics, independent of the AVR).

AVR block (PART 3) differs from the previous project in one structural
way worth flagging: FEX(x) -- the rectifier regulation curve -- is a
genuine 4-branch piecewise function of an operating-point-dependent
variable (x = KC*Ifd_m/Vt_m), not a single-sided clip/limiter. Rather
than fight sympy's Piecewise through the staged-diff machinery, each
branch's closed form is differentiated separately at import time (4
small, cheap diffs), and the ACTIVE branch is selected NUMERICALLY at
evaluation time from x0's value -- exactly the same pattern already
used here for Sat_prime (different closed form per saturation TYPE,
selected via a plain if/elif on which case applies at the OP).

Output: a `linearize_at_operating_point(p, avr_p, Re_total, Xe_total,
R1, X1, Vt0, P0, Q0, sat_S10, sat_S12, sat_kind)` function returning
numeric (A, B, C, D) matrices, obtained by (1) running the EXISTING,
validated numerical initialisation to get a self-consistent operating
point, then (2) evaluating the symbolic Jacobian expressions there. The
symbolic derivation itself is done ONCE at import time; each call is
just numeric substitution (fast).
"""
import time
import numpy as np
import sympy as sp

import genqec_model as gqc
import controls as ctrl

# =======================================================================
# PART 1: network + saturation algebraic loop (verified in
# linearize_step1_network.py -- reproduced here unchanged).
# =======================================================================
Xdpp_sat, Xqpp_sat, RaS, Gline, Bline = sp.symbols('Xdpp_sat Xqpp_sat RaS Gline Bline', real=True)
b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4', real=True)
_A_mat = sp.Matrix([[Xdpp_sat, RaS, 0, 1], [RaS, -Xqpp_sat, 1, 0],
                     [1, 0, -Gline, Bline], [0, 1, -Bline, -Gline]])
_b_vec = sp.Matrix([b1, b2, b3, b4])
_sol = _A_mat.inv() * _b_vec
Id_e, Iq_e, Vd_e, Vq_e = _sol[0], _sol[1], _sol[2], _sol[3]
_abstract_syms = [Xdpp_sat, Xqpp_sat, RaS, Gline, Bline, b1, b2, b3, b4]
_partials = {(nm, s): sp.diff(ex, s) for nm, ex in
             [('Id', Id_e), ('Iq', Iq_e), ('Vd', Vd_e), ('Vq', Vq_e)] for s in _abstract_syms}

Eqp, psidp, Edp, psiqp, delta, omega = sp.symbols('Eqp psidp Edp psiqp delta omega', real=True)
Xdpp, Xqpp, Xdp, Xqp, Xl, Ra = sp.symbols('Xdpp Xqpp Xdp Xqp Xl Ra', positive=True)
Xd, Xq, Kw = sp.symbols('Xd Xq Kw', real=True)
Re, Xe, Einf, Sa = sp.symbols('Re Xe Einf Sa', real=True)
Sat_prime = sp.symbols('Sat_prime', real=True)

Eqpp_e = Eqp * (Xdpp - Xl) / (Xdp - Xl) + psidp * (Xdp - Xdpp) / (Xdp - Xl)
Edpp_e = Edp * (Xqpp - Xl) / (Xqp - Xl) + psiqp * (Xqp - Xqpp) / (Xqp - Xl)
Xdpp_sat_e = Xl + (Xdpp - Xl) / (1 + Sa)
Xqpp_sat_e = Xl + (Xqpp - Xl) / (1 + Sa)
Gline_e = Re / (Re**2 + Xe**2)
Bline_e = -Xe / (Re**2 + Xe**2)
cosd, sind = sp.cos(delta), sp.sin(delta)
b1_e = Eqpp_e * (1 + omega)
b2_e = Edpp_e * (1 + omega)
b3_e = Einf * (Bline_e * cosd - Gline_e * sind)
b4_e = -Einf * (Bline_e * sind + Gline_e * cosd)
_outer = {Xdpp_sat: Xdpp_sat_e, Xqpp_sat: Xqpp_sat_e, RaS: Ra, Gline: Gline_e,
          Bline: Bline_e, b1: b1_e, b2: b2_e, b3: b3_e, b4: b4_e}

state_vars = [Eqp, psidp, Edp, psiqp, delta, omega]
_outer_d_state = {(s, v): sp.diff(ex, v) for s, ex in _outer.items() for v in state_vars}
_outer_d_Sa = {s: sp.diff(ex, Sa) for s, ex in _outer.items()}


def _chain_state(out_name, v):
    return sum(_partials[(out_name, s)] * _outer_d_state.get((s, v), 0) for s in _abstract_syms)


def _chain_Sa(out_name):
    return sum(_partials[(out_name, s)] * _outer_d_Sa.get(s, 0) for s in _abstract_syms)


d_out_d_state_fixedSa = {(nm, v): _chain_state(nm, v) for nm in ['Id', 'Iq', 'Vd', 'Vq'] for v in state_vars}
d_out_d_Sa = {nm: _chain_Sa(nm) for nm in ['Id', 'Iq', 'Vd', 'Vq']}

Vqag_e = Vq_e + Iq_e * RaS + Id_e * Xl
Vdag_e = Vd_e + Id_e * RaS - Iq_e * Xl
u_e = Vqag_e**2 + Vdag_e**2
_du_d_abstract = {s: sp.diff(u_e, s) for s in _abstract_syms}
_du_d_Xl_direct = sp.diff(u_e, Xl)


def _du_d_state(v):
    total = sum(_du_d_abstract[s] * _outer_d_state.get((s, v), 0) for s in _abstract_syms)
    if v is Xl:
        total = total + _du_d_Xl_direct
    return total


def _du_dSa():
    return sum(_du_d_abstract[s] * _outer_d_Sa.get(s, 0) for s in _abstract_syms)


print("[linearize_condenser] Part 1 (network+saturation) symbolic setup done.")

# =======================================================================
# PART 2: machine block (6 states), in terms of ABSTRACT Id,Iq,Sa,Efd
# (their true state-dependence is folded in numerically at evaluation
# time via the Part-1 machinery -- exactly the same staging technique).
# =======================================================================
Id_s, Iq_s, Sa_s, Efd_s = sp.symbols('Id_s Iq_s Sa_s Efd_s', real=True)
Tdop, Tdopp, Tqop, Tqopp, H, D = sp.symbols('Tdop Tdopp Tqop Tqopp H D', positive=True)
Pmech, omega0c = sp.symbols('Pmech omega0c', real=True)
Kidw_s = sp.symbols('Kidw_s', real=True)  # = clip(Kw*Id,-0.25,0.25); handled numerically (piecewise)

Eqpp_m = Eqp * (Xdpp - Xl) / (Xdp - Xl) + psidp * (Xdp - Xdpp) / (Xdp - Xl)
Edpp_m = Edp * (Xqpp - Xl) / (Xqp - Xl) + psiqp * (Xqp - Xqpp) / (Xqp - Xl)
Xdpp_sat_m = Xl + (Xdpp - Xl) / (1 + Sa_s)
Xqpp_sat_m = Xl + (Xqpp - Xl) / (1 + Sa_s)
psi_d_m = Eqpp_m - Id_s * Xdpp_sat_m
psi_q_m = -Edpp_m - Iq_s * Xqpp_sat_m
Telec_m = psi_d_m * Iq_s - psi_q_m * Id_s

ddelta_m = omega * omega0c
domega_m = (Pmech - D * omega) / (1 + omega) / (2 * H) - Telec_m / (2 * H)

LadIfd_m = (1 + Sa_s) / (1 - Kidw_s) * (
    Eqp + (Xd - Xdp) * Id_s / (1 + Sa_s)
    + (Xdp - Xdpp) / (Xdp - Xl) ** 2 * (Eqp - psidp - (Xdp - Xl) * Id_s / (1 + Sa_s)))
dEqp_m = (Efd_s - LadIfd_m) / Tdop
dpsidp_m = ((1 + Sa_s) * (-psidp - (Xdp - Xl) * Id_s / (1 + Sa_s) + Eqp)) / Tdopp
dpsiqp_m = ((1 + Sa_s) * (-psiqp + (Xqp - Xl) * Iq_s / (1 + Sa_s) + Edp)) / Tqopp
dEdp_m = ((1 + Sa_s) * (-Edp + (Xq - Xqp) * Iq_s / (1 + Sa_s)
           - (Xqp - Xqpp) / (Xqp - Xl) ** 2
           * (Edp - psiqp + (Xqp - Xl) * Iq_s / (1 + Sa_s)))) / Tqop

machine_eqs = [ddelta_m, domega_m, dEqp_m, dpsidp_m, dEdp_m, dpsiqp_m]
machine_state_syms = [delta, omega, Eqp, psidp, Edp, psiqp]  # NOTE: order matches genqec_model x6

# d(machine_eq)/d(machine_state), HOLDING Id_s,Iq_s,Sa_s,Efd_s fixed (direct part)
d_machine_d_state_direct = [[sp.diff(eq, v) for v in machine_state_syms] for eq in machine_eqs]
# d(machine_eq)/d(Id_s,Iq_s,Sa_s,Efd_s)
algebraic_inputs = [Id_s, Iq_s, Sa_s, Efd_s]
d_machine_d_alg = [[sp.diff(eq, s) for s in algebraic_inputs] for eq in machine_eqs]
# Kidw_s = clip(Kw*Id,-0.25,0.25) was held as an independent symbol above (only
# appears in dEqp_m, via LadIfd_m's "1/(1-Kidw_s)" factor), but it is ITSELF a
# function of Id_s -- chain rule needs d(machine_eq)/d(Kidw_s) too, combined at
# evaluation time with dKidw/dId (=Kw if interior to +-0.25, else 0).
d_machine_d_Kidw = [sp.diff(eq, Kidw_s) for eq in machine_eqs]
print("[linearize_condenser] Part 2 (machine) symbolic setup done.")

# =======================================================================
# PART 3: AVR block (3 states): Vt_m, Ifd_m, x_I
# =======================================================================
Vt_m, Ifd_m, x_I = sp.symbols('Vt_m Ifd_m x_I', real=True)
Vref_s, Vt_in_s, Ifd_s = sp.symbols('Vref_s Vt_in_s Ifd_s', real=True)
Kp_s, Ki_s, KC, FMAX_s, Tmeas = sp.symbols('Kp_s Ki_s KC FMAX_s Tmeas', real=True)

# ---- the 3 STATE equations. CORRECTED (see chat): there is only ONE
# lag time constant (Tmeas), applied separately to Vt and to Ifd -- NOT
# two lags in series on Vt only, as the previous version had it. Both
# Vt_in_s (raw generator Vt) and Ifd_s (raw machine field current) are
# abstract algebraic inputs, chain-ruled in at evaluation time via
# dVtgen()/dIfd() respectively -- dIfd() already existed (built for the
# machine block's own Efd term originally), reused here for a new
# purpose: it is the same total derivative of the SAME abstract Ifd_s
# symbol, just now needed by a different equation.
dVt_m_avr = (Vt_in_s - Vt_m) / Tmeas
dIfd_m_avr = (Ifd_s - Ifd_m) / Tmeas
e_avr = Vref_s - Vt_m
dx_I_avr = e_avr   # assume interior to [U_MIN,U_MAX] at the OP (checked
                    # numerically -- and guaranteed by avr_initialize,
                    # which raises if it isn't)

avr_eqs = [dVt_m_avr, dIfd_m_avr, dx_I_avr]
avr_state_syms = [Vt_m, Ifd_m, x_I]
avr_inputs = [Vref_s, Vt_in_s, Ifd_s]

d_avr_d_avrstate = [[sp.diff(eq, v) for v in avr_state_syms] for eq in avr_eqs]
d_avr_d_inputs = [[sp.diff(eq, s) for s in avr_inputs] for eq in avr_eqs]

# ---- Efd: a separate OUTPUT expression (like the old Efd_avr). Note
# the structural simplification vs. the previous (incorrect) version:
# Efd is now a function of avr_state_syms ONLY (Vt_m, Ifd_m, x_I) plus
# Vref_s directly -- it has NO direct algebraic dependence on Ifd_s
# itself (that only drives the Ifd_m state equation above, a genuinely
# dynamic/lagged path, not an instantaneous one). u_pi is assumed
# interior to [U_MIN,U_MAX] (guaranteed by avr_initialize, as above).
u_pi_avr = Kp_s * e_avr + Ki_s * x_I
x_fex_avr = KC * Ifd_m / Vt_m

# 4 branch candidates for FEX(x_fex_avr) -- literal decimals matching
# what was specified (0.577/0.433/0.75/1.732), NOT "cleaned up" to their
# nearby exact irrational forms (1/sqrt(3) etc.), so the model matches
# the AVR's actual specified behaviour exactly, not an idealisation of it.
FEX_branches_sym = [1 - 0.577 * x_fex_avr,
                     sp.sqrt(0.75 - x_fex_avr ** 2),
                     1.732 * (1 - x_fex_avr),
                     sp.Integer(0)]
factor_branches_sym = [Vt_m * fb for fb in FEX_branches_sym]
Efd_branches_sym = [u_pi_avr * fac for fac in factor_branches_sym]
# 5th case: factor pinned at its own ceiling (FACTOR_MAX) regardless of
# which FEX branch got it there -- a genuine structural switch (like
# the old field-current limiter), not just "which smooth formula".
Efd_clamped_sym = u_pi_avr * FMAX_s

Efd_d_avrstate_branches = [[sp.diff(Efdk, v) for v in avr_state_syms] for Efdk in Efd_branches_sym]
Efd_d_Vref_branches = [sp.diff(Efdk, Vref_s) for Efdk in Efd_branches_sym]
Efd_d_avrstate_clamped = [sp.diff(Efd_clamped_sym, v) for v in avr_state_syms]
Efd_d_Vref_clamped = sp.diff(Efd_clamped_sym, Vref_s)
print("[linearize_condenser] Part 3 (AVR) symbolic setup done.")

# PART 4 (PSS) removed -- out of scope for this project (disabled, not
# just zeroed). n is fixed at 9 (6 machine + 3 AVR) throughout.

# =======================================================================
# PART 5: Ifd as a function of (Eqp,psidp,Id,Sa). (Vc/load-compensation
# removed -- this AVR uses Vt directly, no Rc/Xc; Vt_generator itself is
# already produced by PART 6 below and reused as the AVR's algebraic
# input via dVtgen(), so no separate Vc_expr is needed here at all. Vd_s,
# Vq_s themselves are still needed -- PART 6 uses them -- so they stay.)
# =======================================================================
Vd_s, Vq_s = sp.symbols('Vd_s Vq_s', real=True)
Kidw_expr = Kw * Id_s   # interior to [-0.25,0.25] assumed (checked numerically)
Ifd_expr = (1 + Sa_s) / (1 - Kidw_expr) * (
    Eqp + (Xd - Xdp) * Id_s / (1 + Sa_s)
    + (Xdp - Xdpp) / (Xdp - Xl) ** 2 * (Eqp - psidp - (Xdp - Xl) * Id_s / (1 + Sa_s)))
Ifd_d_Eqp = sp.diff(Ifd_expr, Eqp)
Ifd_d_psidp = sp.diff(Ifd_expr, psidp)
Ifd_d_Id = sp.diff(Ifd_expr, Id_s)
Ifd_d_Sa = sp.diff(Ifd_expr, Sa_s)
print("[linearize_condenser] Part 5 (Ifd) symbolic setup done.")

# =======================================================================
# PART 6: delivery-bus outputs + generator terminal voltage. Delivery-bus
# quantities now use NET current (machine current minus the local load's
# own draw at the SAME bus) -- see chat: load hangs directly off the
# generator terminal, so only Id_net,Iq_net actually continue on toward
# R1+jX1 and the wider grid. Vt_generator is unaffected (it's a voltage,
# not a current-carrying quantity, at that same bus).
# =======================================================================
R1s, X1s = sp.symbols('R1s X1s', real=True)
Gload_s, Bload_s = sp.symbols('Gload_s Bload_s', real=True)
Id_load_e = Gload_s * Vd_s - Bload_s * Vq_s
Iq_load_e = Gload_s * Vq_s + Bload_s * Vd_s
Id_net_e = Id_s - Id_load_e
Iq_net_e = Iq_s - Iq_load_e

Vd_deliv_e = Vd_s - R1s * Id_net_e + X1s * Iq_net_e
Vq_deliv_e = Vq_s - R1s * Iq_net_e - X1s * Id_net_e
V_deliv_e = sp.sqrt(Vd_deliv_e ** 2 + Vq_deliv_e ** 2)
Q_deliv_e = Vq_deliv_e * Id_net_e - Vd_deliv_e * Iq_net_e
Vt_gen_e = sp.sqrt(Vd_s ** 2 + Vq_s ** 2)   # generator terminal voltage, no Rc/Xc compensation

OUTPUT_EXPRS = (V_deliv_e, Q_deliv_e, Vt_gen_e)
out_d_Vd = [sp.diff(e, Vd_s) for e in OUTPUT_EXPRS]
out_d_Vq = [sp.diff(e, Vq_s) for e in OUTPUT_EXPRS]
out_d_Id = [sp.diff(e, Id_s) for e in OUTPUT_EXPRS]
out_d_Iq = [sp.diff(e, Iq_s) for e in OUTPUT_EXPRS]
print("[linearize_condenser] Part 6 (delivery bus + Vt_generator outputs) symbolic setup done.\n")

# =======================================================================
# PART 7: delivery-point measurement lags (2 states): Vdeliv_m, Qdeliv_m.
# CORRECTED per chat: it's not that these are read instantly -- they go
# through their OWN 0.08s lag each (the constant originally, and
# incorrectly, placed inside the AVR -- see controls.py history). Their
# algebraic inputs (Vdeliv_raw_s, Qdeliv_raw_s) are abstract placeholders
# for V_deliv_e, Q_deliv_e (Part 6, already load-corrected); chain-ruled
# in at evaluation time via dVdeliv()/dQdeliv(), built the same way as
# dVtgen() reuses the Vt_generator output partials.
# =======================================================================
Vdeliv_m, Qdeliv_m = sp.symbols('Vdeliv_m Qdeliv_m', real=True)
Vdeliv_raw_s, Qdeliv_raw_s = sp.symbols('Vdeliv_raw_s Qdeliv_raw_s', real=True)
Tdeliv = sp.symbols('Tdeliv', real=True)

dVdeliv_m_avr = (Vdeliv_raw_s - Vdeliv_m) / Tdeliv
dQdeliv_m_avr = (Qdeliv_raw_s - Qdeliv_m) / Tdeliv
deliv_eqs = [dVdeliv_m_avr, dQdeliv_m_avr]
deliv_state_syms = [Vdeliv_m, Qdeliv_m]
deliv_inputs = [Vdeliv_raw_s, Qdeliv_raw_s]
d_deliv_d_delivstate = [[sp.diff(eq, v) for v in deliv_state_syms] for eq in deliv_eqs]
d_deliv_d_inputs = [[sp.diff(eq, s) for s in deliv_inputs] for eq in deliv_eqs]
print("[linearize_condenser] Part 7 (delivery-point measurement lags) symbolic setup done.")

print("All symbolic setup complete. Call linearize_at_operating_point(...) to evaluate numerically.")

# =======================================================================
# NUMERIC EVALUATION
# =======================================================================
FULL_STATE_SYMS = [delta, omega, Eqp, psidp, Edp, psiqp, Vt_m, Ifd_m, x_I, Vdeliv_m, Qdeliv_m]
STATE_NAMES = ["delta", "omega", "Eqp", "psidp", "Edp", "psiqp",
               "Vt_m", "Ifd_m", "x_I", "Vdeliv_m", "Qdeliv_m"]
OUTPUT_NAMES = ["Vdelivery", "Q_delivered", "Vt_generator", "Vdeliv_m", "Qdeliv_m"]
N_STATES = 11


def _ev(expr, subs):
    """Evaluate a sympy expression to a float given a full substitution dict."""
    return float(expr.subs(subs))


def linearize_at_operating_point(p: 'gqc.GENQECParams', avr_p: 'ctrl.AVRParams',
                                  Re_total, Xe_total, R1, X1, Vt0, P0, Q0,
                                  sat_S10, sat_S12, sat_kind='quadratic',
                                  Gload=0.0, Bload=0.0, Tdeliv_val=0.08):
    """
    Returns dict(A, B, C, D, state_names, output_names, operating_point,
    flags) where A is (11x11), B is (11x1), C is (5x11), D is (5x1)
    zeros. `flags` reports which nonlinearities were interior (valid
    linearization) vs active at this OP -- see module docstring.

    Gload, Bload: local load shunt admittance at the machine terminal
        (see chat / genqec_model.solve_network docstring) -- computed by
        the caller from S_load, cos(phi) at nominal voltage (constant-
        impedance load). Gload=Bload=0 recovers the no-load network.
    Tdeliv_val: the delivery-point measurement lag (Vdeliv_m, Qdeliv_m).
    """
    t_start = time.time()
    state0, Efd0, Pmech0, Einf_val = gqc.initialize(p, Vt0, P0, Q0, Re_total, Xe_total,
                                                      Gf=Gload, Bf=Bload)
    delta0, omega0, Eqp0, psidp0, Edp0, psiqp0 = state0
    Id0, Iq0, Vd0, Vq0, Sa0, psi_ag0 = gqc.solve_network(*state0, p, Einf_val, Re_total, Xe_total,
                                                           Gf=Gload, Bf=Bload)
    Vtgen0 = float(np.hypot(Vd0, Vq0))    # raw generator Vt, no compensation -- this AVR's input
    Ifd0 = gqc.field_current(Eqp0, psidp0, Id0, Sa0, p)
    avr_x0, Vref0 = ctrl.avr_initialize(Vtgen0, Efd0, Ifd0, avr_p)
    Vt_m0, Ifd_m0, x_I0 = avr_x0

    # ---- load-net current + delivery-bus quantities at the OP (closed
    # form, same formulas as Parts 6-7, evaluated numerically here to
    # initialise the 2 new lag states at their true equilibrium) ----
    Id_load0 = Gload * Vd0 - Bload * Vq0
    Iq_load0 = Gload * Vq0 + Bload * Vd0
    Id_net0, Iq_net0 = Id0 - Id_load0, Iq0 - Iq_load0
    Vd_deliv0 = Vd0 - R1 * Id_net0 + X1 * Iq_net0
    Vq_deliv0 = Vq0 - R1 * Iq_net0 - X1 * Id_net0
    Vdeliv_raw0 = float(np.hypot(Vd_deliv0, Vq_deliv0))
    Qdeliv_raw0 = float(Vq_deliv0 * Id_net0 - Vd_deliv0 * Iq_net0)
    Vdeliv_m0, Qdeliv_m0 = Vdeliv_raw0, Qdeliv_raw0    # lag settles to its input at equilibrium
    n = N_STATES

    # ---- Sat'(psi_ag0), by saturation type (closed form) ----
    sat_obj = gqc.make_saturation(sat_kind, sat_S10, sat_S12)
    A_c, B_c = sat_obj.A, sat_obj.B
    if sat_kind == 'quadratic':
        Sat_prime_val = 2 * B_c * (psi_ag0 - A_c) if psi_ag0 > A_c else 0.0
    elif sat_kind == 'scaled_quadratic':
        Sat_prime_val = B_c * (psi_ag0 ** 2 - A_c ** 2) / psi_ag0 ** 2 if psi_ag0 > A_c else 0.0
    elif sat_kind == 'exponential':
        Sat_prime_val = A_c * B_c * psi_ag0 ** (A_c - 1)
    else:
        raise ValueError(sat_kind)

    # ---- AVR nonlinearity flags (linearization validity at this OP) ----
    flags = {}
    flags['Kidw_clamped'] = abs(p.Kw * Id0) >= 0.25
    u_pi0 = avr_p.Ki * x_I0   # = Kp*e0 + Ki*x_I0, e0=0 at equilibrium
    flags['PI_clamped'] = not (avr_p.U_MIN < u_pi0 < avr_p.U_MAX)  # guaranteed False if avr_initialize succeeded
    x0_fex = avr_p.KC * Ifd_m0 / Vt_m0
    fex0 = ctrl._fex(x0_fex)
    flags['factor_clamped'] = (Vt_m0 * fex0) >= avr_p.FACTOR_MAX
    if x0_fex <= 0.433:
        branch_idx = 0
    elif x0_fex <= 0.75:
        branch_idx = 1
    elif x0_fex <= 1.0:
        branch_idx = 2
    else:
        branch_idx = 3
    flags['fex_branch'] = branch_idx + 1   # 1-indexed, informational
    flags['fex_near_kink'] = abs(x0_fex - 1.0) < 0.02   # the ONE non-smooth (C0-only) transition, at x=1
    flags['Vref_clamped'] = Vref0 >= avr_p.VREF_MAX   # guaranteed False if avr_initialize succeeded
    any_active = (flags['Kidw_clamped'] or flags['PI_clamped'] or flags['factor_clamped']
                  or flags['fex_near_kink'] or flags['Vref_clamped'])

    # ---- substitution dict: ALL parameters + operating point ----
    subs = {
        Xdpp: p.Xdpp, Xqpp: p.Xqpp, Xdp: p.Xdp, Xqp: p.Xqp, Xl: p.Xl, Ra: p.Ra,
        Xd: p.Xd, Xq: p.Xq, Kw: p.Kw,
        Re: Re_total, Xe: Xe_total, Einf: Einf_val,
        Tdop: p.Tdop, Tdopp: p.Tdopp, Tqop: p.Tqop, Tqopp: p.Tqopp, H: p.H, D: p.D,
        Pmech: Pmech0, omega0c: p.omega0,
        Eqp: Eqp0, psidp: psidp0, Edp: Edp0, psiqp: psiqp0, delta: delta0, omega: omega0,
        Sa: Sa0, Id_s: Id0, Iq_s: Iq0, Vd_s: Vd0, Vq_s: Vq0, Sat_prime: Sat_prime_val,
        Sa_s: Sa0,
        Efd_s: Efd0, Vref_s: Vref0, Vt_in_s: Vtgen0, Ifd_s: Ifd0,
        Vt_m: Vt_m0, Ifd_m: Ifd_m0, x_I: x_I0,
        Kp_s: avr_p.Kp, Ki_s: avr_p.Ki, KC: avr_p.KC, FMAX_s: avr_p.FACTOR_MAX,
        Tmeas: avr_p.T_MEAS,
        R1s: R1, X1s: X1, Gload_s: Gload, Bload_s: Bload,
        Vdeliv_m: Vdeliv_m0, Qdeliv_m: Qdeliv_m0, Tdeliv: Tdeliv_val,
    }
    Kidw_val = np.clip(p.Kw * Id0, -0.25, 0.25)
    subs[Kidw_s] = Kidw_val

    # abstract intermediate symbols (Xdpp_sat, Gline, etc.) need numeric
    # values too -- everything in Part 1 is expressed in terms of them
    abstract_vals = {s: float(_outer[s].subs(subs)) for s in _abstract_syms}
    # local load = extra shunt admittance in PARALLEL with the line, at
    # the SAME machine-terminal node -- adds directly onto Gline,Bline
    # (both are already state-independent, so this is exact, not an
    # approximation; see chat/README). Does NOT touch b3,b4 (the
    # infinite-bus injection) -- the load isn't a source.
    abstract_vals[Gline] += Gload
    abstract_vals[Bline] += Bload
    subs.update(abstract_vals)

    # ---- Part 1: dId,dIq,dVd,dVq,dSa / d(6 machine states) ----
    dSa_dstate = {}
    du_dSa_val = _ev(_du_dSa(), subs)
    sqrt_u0 = psi_ag0 * (1 + omega0)  # psi_ag = sqrt(u)/(1+omega), and psi_ag0 already known exactly
    dpsiag_dSa_val = du_dSa_val / (2 * sqrt_u0) / (1 + omega0)
    denom0 = 1 - Sat_prime_val * dpsiag_dSa_val
    for v in state_vars:
        du_dv_val = _ev(_du_d_state(v), subs)
        dpsiag_dv_val = du_dv_val / (2 * sqrt_u0) / (1 + omega0)
        if v is omega:
            dpsiag_dv_val += -sqrt_u0 / (1 + omega0) ** 2
        dSa_dstate[v] = Sat_prime_val * dpsiag_dv_val / denom0

    d_alg_d_state = {}  # d_alg_d_state[('Id',v)] for v in state_vars (the 6 machine-state symbols)
    for name in ['Id', 'Iq', 'Vd', 'Vq']:
        dout_dSa_val = _ev(d_out_d_Sa[name], subs)
        for v in state_vars:
            dv_fixed = _ev(d_out_d_state_fixedSa[(name, v)], subs)
            d_alg_d_state[(name, v)] = dv_fixed + dout_dSa_val * dSa_dstate[v]
    d_alg_d_state_full = {'Sa': dSa_dstate, 'Id': {v: d_alg_d_state[('Id', v)] for v in state_vars},
                           'Iq': {v: d_alg_d_state[('Iq', v)] for v in state_vars},
                           'Vd': {v: d_alg_d_state[('Vd', v)] for v in state_vars},
                           'Vq': {v: d_alg_d_state[('Vq', v)] for v in state_vars}}

    def dalg(name, sym):
        """d(Id|Iq|Vd|Vq|Sa)/d(sym), 0 if sym is not one of the 6 machine states."""
        return d_alg_d_state_full[name].get(sym, 0.0)

    # ---- output partials (needed both for dVtgen() below and for C) ----
    out_d_Vd_v = [_ev(e, subs) for e in out_d_Vd]
    out_d_Vq_v = [_ev(e, subs) for e in out_d_Vq]
    out_d_Id_v = [_ev(e, subs) for e in out_d_Id]
    out_d_Iq_v = [_ev(e, subs) for e in out_d_Iq]

    def dVtgen(sym):
        """d(Vt_generator)/d(sym) -- reuses the Vt_generator OUTPUT partials
        (index 2 of OUTPUT_EXPRS) as the AVR's algebraic input, exactly as
        dVc() chain-ruled the old load-compensated Vc into the ST1A AVR."""
        return out_d_Vd_v[2] * dalg('Vd', sym) + out_d_Vq_v[2] * dalg('Vq', sym)

    def dVdeliv(sym):
        """d(Vdelivery)/d(sym) -- reuses the Vdelivery OUTPUT partials
        (index 0), now load-corrected (Part 6), as PART 7's algebraic
        input. Same pattern as dVtgen()/dVc()."""
        return (out_d_Vd_v[0] * dalg('Vd', sym) + out_d_Vq_v[0] * dalg('Vq', sym)
                + out_d_Id_v[0] * dalg('Id', sym) + out_d_Iq_v[0] * dalg('Iq', sym))

    def dQdeliv(sym):
        """d(Q_delivered)/d(sym) -- same pattern, index 1."""
        return (out_d_Vd_v[1] * dalg('Vd', sym) + out_d_Vq_v[1] * dalg('Vq', sym)
                + out_d_Id_v[1] * dalg('Id', sym) + out_d_Iq_v[1] * dalg('Iq', sym))

    # ---- Ifd total derivative w.r.t. the 6 machine states. Previously
    # this fed Efd's own chain rule; now (corrected AVR) it instead feeds
    # the AVR block's dIfd_m_avr row below -- same helper, new use. ----
    Ifd_d_Eqp_v = _ev(Ifd_d_Eqp, subs); Ifd_d_psidp_v = _ev(Ifd_d_psidp, subs)
    Ifd_d_Id_v = _ev(Ifd_d_Id, subs); Ifd_d_Sa_v = _ev(Ifd_d_Sa, subs)

    def dIfd(sym):
        direct = (Ifd_d_Eqp_v if sym is Eqp else 0.0) + (Ifd_d_psidp_v if sym is psidp else 0.0)
        return direct + Ifd_d_Id_v * dalg('Id', sym) + Ifd_d_Sa_v * dalg('Sa', sym)

    # ---- Efd total derivative: branch/clamp-selected at THIS OP. Efd is
    # now a function of avr_state_syms and Vref_s ONLY (see Part 3 note)
    # -- no more indirect Ifd_s chain-rule term needed here at all.
    if flags['factor_clamped']:
        Efd_d_avrstate_v = [_ev(e, subs) for e in Efd_d_avrstate_clamped]
        dEfd_dVref_v = _ev(Efd_d_Vref_clamped, subs)
    else:
        Efd_d_avrstate_v = [_ev(e, subs) for e in Efd_d_avrstate_branches[branch_idx]]
        dEfd_dVref_v = _ev(Efd_d_Vref_branches[branch_idx], subs)
    # VREF_MAX clamp (see chat): if active, Vref_applied is pinned at a
    # constant, so its derivative w.r.t. the TRUE Vref input is 0 --
    # zeroes every Vref-column entry that would otherwise follow.
    vref_gate = 0.0 if flags['Vref_clamped'] else 1.0
    dEfd_dVref_v *= vref_gate

    def dEfd(sym):
        return Efd_d_avrstate_v[avr_state_syms.index(sym)] if sym in avr_state_syms else 0.0

    # ---- assemble full A, B matrices ----
    Anum = np.zeros((n, n))
    Bnum = np.zeros((n, 1))
    rowsyms = FULL_STATE_SYMS

    d_mach_direct_v = [[_ev(e, subs) for e in row] for row in d_machine_d_state_direct]
    d_mach_alg_v = [[_ev(e, subs) for e in row] for row in d_machine_d_alg]  # [Id,Iq,Sa,Efd] per eq
    d_mach_Kidw_v = [_ev(e, subs) for e in d_machine_d_Kidw]
    dKidw_dId_val = 0.0 if flags['Kidw_clamped'] else p.Kw
    for i in range(6):
        for j, sym in enumerate(rowsyms):
            direct = d_mach_direct_v[i][machine_state_syms.index(sym)] if sym in machine_state_syms else 0.0
            alg = (d_mach_alg_v[i][0] * dalg('Id', sym) + d_mach_alg_v[i][1] * dalg('Iq', sym)
                   + d_mach_alg_v[i][2] * dalg('Sa', sym) + d_mach_alg_v[i][3] * dEfd(sym)
                   + d_mach_Kidw_v[i] * dKidw_dId_val * dalg('Id', sym))
            Anum[i, j] = direct + alg
        # This PI has a proportional term, so Efd depends DIRECTLY on
        # Vref (not just state-mediated) -- machine rows pick up a
        # genuine B-column contribution here (see chat/README).
        Bnum[i, 0] = d_mach_alg_v[i][3] * dEfd_dVref_v

    d_avr_direct_v = [[_ev(e, subs) for e in row] for row in d_avr_d_avrstate]
    d_avr_inputs_v = [[_ev(e, subs) for e in row] for row in d_avr_d_inputs]  # [Vref,Vt_in,Ifd] per eq
    for i in range(3):
        row = 6 + i
        for j, sym in enumerate(rowsyms):
            direct = d_avr_direct_v[i][avr_state_syms.index(sym)] if sym in avr_state_syms else 0.0
            # index1=Vt_in_s (nonzero only for the Vt_m row), index2=Ifd_s
            # (nonzero only for the Ifd_m row) -- sympy already zeroed the
            # other row's coefficient, so this sum is safe to apply uniformly.
            alg = d_avr_inputs_v[i][1] * dVtgen(sym) + d_avr_inputs_v[i][2] * dIfd(sym)
            Anum[row, j] = direct + alg
        Bnum[row, 0] = d_avr_inputs_v[i][0] * vref_gate   # d(avr_eq)/dVref, direct (only dx_I row is nonzero)

    # ---- PART 7 rows: delivery-point measurement lags (Vdeliv_m,
    # Qdeliv_m) -- no Vref dependence at all (B stays 0 for these rows).
    d_deliv_direct_v = [[_ev(e, subs) for e in row] for row in d_deliv_d_delivstate]
    d_deliv_inputs_v = [[_ev(e, subs) for e in row] for row in d_deliv_d_inputs]  # [Vdeliv_raw,Qdeliv_raw] per eq
    deliv_chain_fns = [dVdeliv, dQdeliv]
    for i in range(2):
        row = 9 + i
        for j, sym in enumerate(rowsyms):
            direct = d_deliv_direct_v[i][deliv_state_syms.index(sym)] if sym in deliv_state_syms else 0.0
            alg = d_deliv_inputs_v[i][i] * deliv_chain_fns[i](sym)   # diagonal: eq i only uses input i
            Anum[row, j] = direct + alg
        # Bnum[row,0] stays 0.0

    # ---- output matrix C (5 x n): Vdelivery, Q_delivered, Vt_generator,
    # Vdeliv_m, Qdeliv_m (the last 2 are now plain states -> identity rows)
    Cnum = np.zeros((5, n))
    for k in range(3):
        for j, sym in enumerate(rowsyms):
            Cnum[k, j] = (out_d_Vd_v[k] * dalg('Vd', sym) + out_d_Vq_v[k] * dalg('Vq', sym)
                          + out_d_Id_v[k] * dalg('Id', sym) + out_d_Iq_v[k] * dalg('Iq', sym))
    Cnum[3, rowsyms.index(Vdeliv_m)] = 1.0
    Cnum[4, rowsyms.index(Qdeliv_m)] = 1.0
    Dnum = np.zeros((5, 1))

    print(f"[linearize_at_operating_point] done in {time.time()-t_start:.3f}s. "
          f"delta0={np.degrees(delta0):.3f}deg, Vref0={Vref0:.5f}, any_flag_active={any_active} {flags}")

    return dict(A=Anum, B=Bnum, C=Cnum, D=Dnum,
                state_names=STATE_NAMES, output_names=OUTPUT_NAMES,
                operating_point=dict(delta0=delta0, omega0=omega0, Eqp0=Eqp0, psidp0=psidp0,
                                      Edp0=Edp0, psiqp0=psiqp0, Id0=Id0, Iq0=Iq0, Vd0=Vd0, Vq0=Vq0,
                                      Sa0=Sa0, psi_ag0=psi_ag0, Efd0=Efd0, Vref0=Vref0, Vtgen0=Vtgen0,
                                      Ifd0=Ifd0, Pmech0=Pmech0, Einf=Einf_val,
                                      Gload=Gload, Bload=Bload, Id_load0=Id_load0, Iq_load0=Iq_load0,
                                      Vdeliv_raw0=Vdeliv_raw0, Qdeliv_raw0=Qdeliv_raw0),
                flags=flags)


if __name__ == "__main__":
    # ILLUSTRATIVE machine parameters (same placeholder set used for this
    # mechanism's own self-test in the previous project) -- NOT the real
    # new machine, EXCEPT H=15/D=0 which ARE real per chat (everything
    # else stays the placeholder set until the real machine is given).
    # AVR parameters ARE the real ones (given, not invented).
    sat = gqc.make_saturation('quadratic', 0.10, 0.30)
    p = gqc.GENQECParams(Ra=0.003, Xl=0.15, Xd=1.80, Xdp=0.30, Xdpp=0.22,
                          Xq=1.70, Xqp=0.50, Xqpp=0.25,
                          Tdop=7.5, Tdopp=0.03, Tqop=0.50, Tqopp=0.05,
                          H=15.0, D=0.0, Kw=0.20, sat=sat, f0=60.0)
    AVR = ctrl.AVRParams()   # real: Kp=72, Ki=35, U=[-2.6,3.25], KC=0.0308, FACTOR_MAX=1.4, T_MEAS=0.0111, VREF_MAX=1.15
    Rt, Xt, Rline, Xline, Rgrid, Xgrid = 0.002, 0.12, 0.01, 0.08, 0.01, 0.06
    Re_total, Xe_total = Rt + Rline + Rgrid, Xt + Xline + Xgrid
    R1, X1 = Rt + Rline, Xt + Xline

    # local load: 8% of condenser nominal, cos(phi)=0.85 lagging, at
    # nominal voltage (constant-impedance load) -- real, per chat.
    S_load, cosphi = 0.08, 0.85
    P_load, Q_load = S_load * cosphi, S_load * (1 - cosphi ** 2) ** 0.5
    Gload, Bload = P_load / 1.0 ** 2, -Q_load / 1.0 ** 2

    result = linearize_at_operating_point(p, AVR, Re_total, Xe_total, R1, X1,
                                           Vt0=1.0, P0=0.0, Q0=0.35,
                                           sat_S10=0.10, sat_S12=0.30, sat_kind='quadratic',
                                           Gload=Gload, Bload=Bload, Tdeliv_val=0.08)
    A, B, C, D = result['A'], result['B'], result['C'], result['D']
    print("\nStates:", result['state_names'])
    print("Outputs:", result['output_names'])
    print("\nEigenvalues of A:")
    for e in sorted(np.linalg.eigvals(A), key=lambda z: -z.real):
        print(f"  {e.real:12.5f} + {e.imag:10.5f}j")

    np.savez("linearized_condenser_model.npz", A=A, B=B, C=C, D=D,
             state_names=result['state_names'], output_names=result['output_names'],
             **{f"op_{k}": v for k, v in result['operating_point'].items()})
    print("\nSaved linearized_condenser_model.npz")
    try:
        from scipy.io import savemat
        savemat("linearized_condenser_model.mat", dict(A=A, B=B, C=C, D=D))
        print("Saved linearized_condenser_model.mat")
    except ImportError:
        pass
