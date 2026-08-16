import sys as _sys, os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ['01_model', '02_linearization']:
    _p = _os.path.join(_PKG_ROOT, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
"""
verify_fex_branches.py
=========================
Focused verification of the ONE genuinely new nonlinearity in this
project's AVR: FEX(x), a 4-branch piecewise function (see controls.py
docstring). linearize_condenser.py picks ONE branch's closed-form
derivative numerically, matching the branch active at a given operating
point (see module docstring there) -- this script checks all 4 branches
directly against finite differences of the real ctrl.avr_derivatives(),
independent of whether any particular machine's achievable operating
range actually reaches each branch.

Efd is a function of the 3 AVR states (Vt_m, Ifd_m, x_I) and Vref ONLY
(see controls.py / linearize_condenser.py Part 3 docstring) -- it does
NOT depend on the raw Vt/Ifd function arguments of avr_derivatives()
directly (those only drive the two lag states' OWN derivatives). So
Efd_num() below passes dummy values for those two arguments.
"""
import numpy as np
import controls as ctrl
import linearize_condenser as lc

AVR = ctrl.AVRParams()
Vref_val = 1.0


def Efd_num(Vt_m, Ifd_m, x_I):
    """Efd as a function of the 3 AVR states alone, at fixed Vref."""
    x = np.array([Vt_m, Ifd_m, x_I])
    _, Efd = ctrl.avr_derivatives(x, Vref_val, Vt_m, Ifd_m, AVR)  # dummy Vt/Ifd args, irrelevant to Efd
    return Efd


# one interior point per branch, x = KC*Ifd_m/Vt_m targeted at each region
test_pts = [(0.20, 'branch1  (x<=0.433)'), (0.55, 'branch2  (0.433<x<=0.75)'),
            (0.90, 'branch3  (0.75<x<=1)'), (1.10, 'branch4  (x>1)')]

h = 1e-6
print(f"{'branch':26s}{'dEfd/dVt_m':>14s}{'fd':>12s}{'dEfd/dIfd_m':>13s}{'fd':>12s}{'dEfd/dx_I':>12s}{'fd':>12s}")
max_err = 0.0
for x_target, label in test_pts:
    Vt_m_0 = 1.0
    Ifd_m_0 = x_target * Vt_m_0 / AVR.KC
    x_I_0 = 0.05   # arbitrary interior x_I (u_pi stays within [U_MIN,U_MAX] since Vref-Vt_m=0 here)

    if x_target <= 0.433: k = 0
    elif x_target <= 0.75: k = 1
    elif x_target <= 1.0: k = 2
    else: k = 3

    subs = {lc.Vref_s: Vref_val, lc.Vt_m: Vt_m_0, lc.Ifd_m: Ifd_m_0, lc.x_I: x_I_0,
            lc.Kp_s: AVR.Kp, lc.Ki_s: AVR.Ki, lc.KC: AVR.KC, lc.FMAX_s: AVR.FACTOR_MAX}
    d_dVtm_sym = float(lc.Efd_d_avrstate_branches[k][0].subs(subs))    # avr_state_syms index0=Vt_m
    d_dIfdm_sym = float(lc.Efd_d_avrstate_branches[k][1].subs(subs))  # index1=Ifd_m
    d_dxI_sym = float(lc.Efd_d_avrstate_branches[k][2].subs(subs))    # index2=x_I

    d_dVtm_fd = (Efd_num(Vt_m_0 + h, Ifd_m_0, x_I_0) - Efd_num(Vt_m_0 - h, Ifd_m_0, x_I_0)) / (2 * h)
    d_dIfdm_fd = (Efd_num(Vt_m_0, Ifd_m_0 + h, x_I_0) - Efd_num(Vt_m_0, Ifd_m_0 - h, x_I_0)) / (2 * h)
    d_dxI_fd = (Efd_num(Vt_m_0, Ifd_m_0, x_I_0 + h) - Efd_num(Vt_m_0, Ifd_m_0, x_I_0 - h)) / (2 * h)

    errs = [abs(d_dVtm_sym - d_dVtm_fd), abs(d_dIfdm_sym - d_dIfdm_fd), abs(d_dxI_sym - d_dxI_fd)]
    max_err = max(max_err, *errs)
    print(f"{label:26s}{d_dVtm_sym:14.6f}{d_dVtm_fd:12.6f}{d_dIfdm_sym:13.6f}{d_dIfdm_fd:12.6f}"
          f"{d_dxI_sym:12.6f}{d_dxI_fd:12.6f}")

print(f"\nMax abs error across all 4 branches x 3 partials: {max_err:.3e}")
print("OVERALL:", "PASS" if max_err < 1e-4 else "FAIL")
