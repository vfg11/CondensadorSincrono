"""
controls.py -- AVR control block for the new machine.

PSS is disabled/out of scope for this project (see project summary) --
not implemented at all here, not just zeroed.

AVR: PI voltage regulator followed by a multiplicative, terminal-
voltage-dependent ceiling (the IEEE Std 421.5-style rectifier
regulation curve, FEX(IN), representing commutation voltage drop in a
potential-source static exciter). No load compensation -- the error
uses Vt (generator terminal voltage) directly, not a Rc/Xc-compensated
Vc.

CORRECTED in chat (previous version had this wrong): there is only ONE
measurement lag, T_MEAS=0.0111s -- the 0.08s constant does not belong
to this AVR at all. That single lag time constant is applied to BOTH
Vt and Ifd, as two SEPARATE first-order lags (not two lags in series
on one signal).

    Vt  --lag(T_MEAS)--> Vt_m
    Ifd --lag(T_MEAS)--> Ifd_m
    e         = Vref - Vt_m
    u_pi      = Kp*e + Ki*x_I                        (x_I = integral of e)
    u_pi_sat  = clip(u_pi, U_MIN, U_MAX)              (clamping/conditional-
                                                        integration anti-
                                                        windup -- same style
                                                        as the previous
                                                        project's ST1A x_A
                                                        state; back-
                                                        calculation was used
                                                        one level up, for the
                                                        outer LQI integral,
                                                        not here)
    x         = KC * Ifd_m / Vt_m
    FEX(x)    = 1 - 0.577*x            x <= 0.433
              = sqrt(0.75 - x^2)       0.433 < x <= 0.75
              = 1.732*(1 - x)          0.75  < x <= 1
              = 0                      x > 1
    factor    = min(Vt_m * FEX(x), FACTOR_MAX)
    Efd       = u_pi_sat * factor                    (no separate exciter
                                                        dynamics -- static/
                                                        instantaneous, same
                                                        assumption as ST1A)

3 states, in order [Vt_m, Ifd_m, x_I] (same count as the previous
version -- one lag stage moved from being a 2nd Vt stage to being its
own Ifd lag, so nothing shrinks or grows here, only the wiring changes).

Ifd (field current, pre-lag) comes from the machine model (GENQEC:
field_current(), exact), same convention as the previous project. Vt
(pre-lag) is the GENERATOR TERMINAL voltage (confirmed in chat).

Each block below exposes the same shape of interface as the previous
project's controls.py:
  - a Params dataclass
  - initialize(...) -> consistent state at a given (constant) operating
    point, so a fault simulation can start from a true equilibrium
  - derivatives(x, u, params) -> (dx/dt, y)   [state derivative + output]
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class AVRParams:
    """PI regulator + IEEE-421.5-style rectifier regulation ceiling.

    Kp, Ki: PI gains (error = Vref - Vt_m).
    U_MIN, U_MAX: PI output saturation / anti-windup limits, applied
        BEFORE multiplication by the ceiling factor (i.e. these bound
        u_pi, not Efd).
    KC: rectifier loading constant (x = KC*Ifd_m/Vt_m).
    FACTOR_MAX: upper limit on the ceiling FACTOR itself (Vt_m*FEX(x)),
        not on Efd directly.
    T_MEAS: the single measurement lag time constant, applied
        separately to both Vt and Ifd (two independent states, same T).
    VREF_MAX: hard ceiling on the reference the AVR will accept, applied
        BEFORE the error is formed (Vref_applied = min(Vref_raw,
        VREF_MAX)) -- confirmed in chat: upper bound only, no stated
        lower bound, so none is applied.
    """
    Kp: float = 72.0
    Ki: float = 35.0
    U_MIN: float = -2.6
    U_MAX: float = 3.25
    KC: float = 0.0308
    FACTOR_MAX: float = 1.4
    T_MEAS: float = 0.0111
    VREF_MAX: float = 1.05   # corregido segun documentacion tecnica revisada -- ver chat
    VREF_MIN: float = 0.95   # corregido segun documentacion tecnica revisada -- ver chat


def _fex(x: float) -> float:
    """IEEE-421.5-style rectifier regulation curve, FEX(IN). Scalar."""
    if x <= 0.433:
        return 1.0 - 0.577 * x
    elif x <= 0.75:
        return float(np.sqrt(max(0.75 - x ** 2, 0.0)))
    elif x <= 1.0:
        return 1.732 * (1.0 - x)
    else:
        return 0.0


def avr_initialize(Vt0: float, Efd0: float, Ifd0: float, p: AVRParams):
    """Consistent steady state for a given (Vt0, Efd0, Ifd0) operating
    point. Returns (x0 = [Vt_m, Ifd_m, x_I], Vref).

    At any equilibrium both lags settle to their input exactly (unity
    DC gain), so Vt_m0=Vt0 and Ifd_m0=Ifd0. dx_I/dt = e = 0 at
    equilibrium too, which forces Vref0 = Vt0 exactly (zero steady-
    state error -- the defining property of integral control): Vref0
    is NOT a free choice here, it falls out of Vt0.
    """
    x0_fex = p.KC * Ifd0 / Vt0
    fex0 = _fex(x0_fex)
    factor0 = min(Vt0 * fex0, p.FACTOR_MAX)
    if factor0 <= 1e-9:
        raise ValueError(
            f"factor0={factor0:.6f} (x={x0_fex:.4f}) -- FEX has collapsed "
            "to ~0 (x > 1: rectifier fully commutation-limited). Ifd0 is "
            "too large relative to Vt0 for this operating point.")
    u_pi0 = Efd0 / factor0
    if not (p.U_MIN < u_pi0 < p.U_MAX):
        raise ValueError(
            f"Required u_pi0={u_pi0:.4f} is outside (U_MIN,U_MAX)="
            f"({p.U_MIN},{p.U_MAX}) -- widen the PI limits or pick a "
            "different operating point.")
    x_I0 = u_pi0 / p.Ki           # e0=0 at equilibrium => u_pi0 = Ki*x_I0
    Vref0 = Vt0                   # e0 = Vref0 - Vt_m0 = 0
    if Vref0 > p.VREF_MAX or Vref0 < p.VREF_MIN:
        raise ValueError(f"Vref0={Vref0:.4f} (=Vt0) fuera de [VREF_MIN,VREF_MAX]="
                          f"({p.VREF_MIN},{p.VREF_MAX}) -- este punto de operacion "
                          "no es alcanzable con este AVR.")
    return np.array([Vt0, Ifd0, x_I0]), Vref0


def avr_derivatives(x, Vref: float, Vt: float, Ifd: float, p: AVRParams):
    """x = [Vt_m, Ifd_m, x_I]. Returns (dx/dt [3], Efd)."""
    Vt_m, Ifd_m, x_I = x

    Vref_applied = float(np.clip(Vref, p.VREF_MIN, p.VREF_MAX))

    dVt_m = (Vt - Vt_m) / p.T_MEAS
    dIfd_m = (Ifd - Ifd_m) / p.T_MEAS

    e = Vref_applied - Vt_m
    u_pi = p.Kp * e + p.Ki * x_I
    u_pi_sat = float(np.clip(u_pi, p.U_MIN, p.U_MAX))
    # Retrocalculo continuo (reemplaza la congelacion condicional anterior
    # -- discontinuidad dura, fragil numericamente con Radau bajo
    # saturacion severa y sostenida; ver chat). Kb=1/Ki, Ki>0 da el signo
    # correcto (al saturar, u_pi_sat-u_pi es negativo, frenando el
    # crecimiento de x_I en la direccion que causa la saturacion).
    Kb = 1.0 / p.Ki
    dx_I = e + Kb * (u_pi_sat - u_pi)

    x_fex = p.KC * Ifd_m / Vt_m
    factor = min(Vt_m * _fex(x_fex), p.FACTOR_MAX)
    Efd = u_pi_sat * factor

    return np.array([dVt_m, dIfd_m, dx_I]), Efd
