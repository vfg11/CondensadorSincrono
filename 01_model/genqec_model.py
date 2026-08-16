"""
GENQEC synchronous machine model
==================================
Implements Sec. 1-1.2 of "GENQEC-Equations.pdf" (J. Weber, PowerWorld).

Unlike GENTPF/GENTPJ, GENQEC applies a SINGLE saturation multiplier
(1+Sa), Sa=Sat(psi_ag), identically to BOTH axes (Sat_d=Sat_q=1+Sa, no
Xq/Xd scaling, no Kis current term) -- but it saturates every reactance
AND every d-axis time constant AND Efd/field-current, not just the
sub-transient network reactances as GENTPF/GENTPJ do. It also keeps the
original GENROU-style states (Eq', psi_d', Ed', psi_q') rather than
GENTPF's relabelled (Eq',Ed',Eq'',Ed''), with psi_d''(=Eq'') and
psi_q''(=-Ed'') as algebraic (flux-divider) outputs -- and adds a
field-current correction Kw (limited to +/-0.25 via Kidw) not present
in GENTPF/GENTPJ.

STATES (6):      delta, omega, Eq', psi_d', Ed', psi_q'
ALGEBRAIC VARS:  Id, Iq, Vdterm, Vqterm, Sa, psi_ag

Every derivative formula below was verified symbolically against the
document's own steady-state equations (Sec. 1.1, items 1-5) before being
coded here -- see verify_genqec.py. The closed-form initial-angle formula
(Sec. 1.1.1) was independently re-derived and matches exactly, and the
whole initialisation turns out to be closed-form (no fsolve needed) since
psi_ag can be evaluated in the network reference frame before delta is
known.

As with GENTPF/GENTPJ, scipy's solve_ivp(method='Radau') has no public
mass-matrix/DAE interface, so the algebraic loop (network Vd,Vq <-> Sa <->
psi_ag) is solved by fixed-point (Picard) iteration to tight tolerance
inside every RHS evaluation, and the resulting reduced ODE is integrated
with Radau -- mathematically the same DAE, solved by nested iteration
instead of a mass-matrix formulation.
"""

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------
# Saturation functions (identical menu to GENTPF/GENTPJ, Sec. 1.2 of the
# earlier document; GENQEC does not redefine them, it reuses Sat(x)).
# ---------------------------------------------------------------------

class SaturationFunction:
    def __call__(self, x):
        return self.value(np.asarray(x, dtype=float))

    def value(self, x):
        raise NotImplementedError


class QuadraticSaturation(SaturationFunction):
    def __init__(self, A, B):
        self.A, self.B = float(A), float(B)

    @classmethod
    def from_S10_S12(cls, S10, S12):
        r = np.sqrt(S12 / S10)
        A = (1.2 - r) / (1.0 - r)
        B = S10 / (1.0 - A) ** 2
        return cls(A, B)

    def value(self, x):
        return np.where(x > self.A, self.B * (x - self.A) ** 2, 0.0)


class ScaledQuadraticSaturation(SaturationFunction):
    def __init__(self, A, B):
        self.A, self.B = float(A), float(B)

    @classmethod
    def from_S10_S12(cls, S10, S12):
        r = np.sqrt(1.2 * S12 / S10)
        A = (1.2 - r) / (1.0 - r)
        B = S10 / (1.0 - A) ** 2
        return cls(A, B)

    def value(self, x):
        xs = np.where(x == 0, 1e-12, x)
        return np.where(x > self.A, self.B * (x - self.A) ** 2 / xs, 0.0)


class ExponentialSaturation(SaturationFunction):
    def __init__(self, A, B):
        self.A, self.B = float(A), float(B)

    @classmethod
    def from_S10_S12(cls, S10, S12):
        A = np.log(S12 / S10) / np.log(1.2)
        return cls(A, S10)

    def value(self, x):
        return self.B * np.maximum(x, 0.0) ** self.A


SATURATION_TYPES = {
    "quadratic": QuadraticSaturation,
    "scaled_quadratic": ScaledQuadraticSaturation,
    "exponential": ExponentialSaturation,
}


def make_saturation(kind, S10, S12):
    if kind not in SATURATION_TYPES:
        raise ValueError(f"kind must be one of {list(SATURATION_TYPES)}")
    return SATURATION_TYPES[kind].from_S10_S12(S10, S12)


# ---------------------------------------------------------------------
# Machine parameters
# ---------------------------------------------------------------------

@dataclass
class GENQECParams:
    """Per-unit machine data (machine base)."""
    Ra: float
    Xl: float
    Xd: float
    Xdp: float       # Xd'
    Xdpp: float      # Xd''
    Xq: float
    Xqp: float       # Xq'
    Xqpp: float      # Xq''
    Tdop: float      # Tdo'  [s]
    Tdopp: float     # Tdo'' [s]
    Tqop: float      # Tqo'  [s]
    Tqopp: float     # Tqo'' [s]
    H: float         # MWs/MVA
    D: float
    Kw: float        # field-current saturation-compensation gain (Kidw limited +-0.25)
    sat: SaturationFunction
    f0: float = 60.0
    poles: int = 2   # nameplate pole count. NOT used in any electrical or
                      # swing-equation term (see gentpj_model.GENTPJParams
                      # for the same note) -- kept purely to convert omega
                      # to actual mechanical RPM for display.

    @property
    def sync_rpm(self):
        """Mechanical synchronous speed [RPM] = 120*f0/poles."""
        return 120.0 * self.f0 / self.poles

    @property
    def omega0(self):
        return 2 * np.pi * self.f0


STATE_NAMES = ["delta", "omega", "Eqp", "psidp", "Edp", "psiqp"]


# ---------------------------------------------------------------------
# Network interface + saturation, solved together by fixed-point
# iteration on the single Sa (Sec. 1, 1.3).
# ---------------------------------------------------------------------

def solve_network(delta, omega, Eqp, psidp, Edp, psiqp, p: GENQECParams,
                   Einf, Re, Xe, Gf=0.0, Bf=0.0, sat_guess=1.0, tol=1e-11, max_iter=100):
    """
    Solves the SMIB network (machine -> Re+jXe -> infinite bus Einf<0deg,
    always present/fixed) together with Sa=Sat(psi_ag) by fixed-point
    iteration.

    Gf, Bf: shunt admittance (conductance, susceptance) at the machine
        terminal [pu], in parallel with the Re+jXe line (Gf=Bf=0, the
        default, recovers the original network exactly). Originally
        just Gf (a fault conductance); generalised to a complex Gf+jBf
        shunt so the SAME mechanism can also represent a permanent,
        non-purely-resistive local load (see chat) -- Gf/Bf is agnostic
        to WHY the shunt is there, only that one is present. A transient
        fault and a permanent load can be summed into the same Gf,Bf by
        the caller if both are ever needed simultaneously.

    Gf=0, the default, recovers the original network exactly -- same
        4x4-reduces-to-2x2 identity verified for GENTPF/GENTPJ in
        verify_arc_network.py applies here unchanged, since it only
        involves the network/saturation block, not the machine's
        internal state equations).

    Returns: Id, Iq, Vdterm, Vqterm, Sa, psi_ag
    """
    # algebraic flux-divider outputs (Derivative Calculations items 1-2)
    Eqpp = Eqp * (p.Xdpp - p.Xl) / (p.Xdp - p.Xl) + psidp * (p.Xdp - p.Xdpp) / (p.Xdp - p.Xl)   # = psi_d''
    Edpp = Edp * (p.Xqpp - p.Xl) / (p.Xqp - p.Xl) + psiqp * (p.Xqp - p.Xqpp) / (p.Xqp - p.Xl)    # = -psi_q''

    Sa = sat_guess
    cosd, sind = np.cos(delta), np.sin(delta)
    denom = Re ** 2 + Xe ** 2
    Gline = Re / denom
    Bline = -Xe / denom

    for _ in range(max_iter):
        Xdpp_sat = p.Xl + (p.Xdpp - p.Xl) / (1 + Sa)
        Xqpp_sat = p.Xl + (p.Xqpp - p.Xl) / (1 + Sa)

        A = np.array([
            [Xdpp_sat, p.Ra, 0.0, 1.0],
            [p.Ra, -Xqpp_sat, 1.0, 0.0],
            [1.0, 0.0, -(Gf + Gline), (Bline + Bf)],
            [0.0, 1.0, -(Bline + Bf), -(Gf + Gline)],
        ])
        b = np.array([
            Eqpp * (1 + omega),
            Edpp * (1 + omega),
            Einf * (Bline * cosd - Gline * sind),
            -Einf * (Bline * sind + Gline * cosd),
        ])
        Id, Iq, Vdterm, Vqterm = np.linalg.solve(A, b)

        Vqag = Vqterm + Iq * p.Ra + Id * p.Xl
        Vdag = Vdterm + Id * p.Ra - Iq * p.Xl
        psi_ag = (1.0 / (1 + omega)) * np.sqrt(Vqag ** 2 + Vdag ** 2)
        Sa_new = float(p.sat(psi_ag))

        if abs(Sa_new - Sa) < tol:
            Sa = Sa_new
            break
        Sa = Sa_new

    return Id, Iq, Vdterm, Vqterm, Sa, psi_ag


# ---------------------------------------------------------------------
# State derivatives -- FULL formulas (items 3-6 of Derivative
# Calculations), not the steady-state-simplified versions.
# ---------------------------------------------------------------------

def field_current(Eqp, psidp, Id, Sa, p: GENQECParams):
    """L_ad*I_fd (Sec 1.2 item 8), exposed standalone so an AVR's field-
    current-dependent limiter terms (KC, KLR/ILR) can use the machine's
    own saturated field current rather than an Efd-based proxy."""
    Kidw = np.clip(p.Kw * Id, -0.25, 0.25)
    return (1 + Sa) / (1 - Kidw) * (
        Eqp + (p.Xd - p.Xdp) * Id / (1 + Sa)
        + (p.Xdp - p.Xdpp) / (p.Xdp - p.Xl) ** 2
        * (Eqp - psidp - (p.Xdp - p.Xl) * Id / (1 + Sa)))


def derivatives(t, x, p: GENQECParams, Efd, Pmech, Einf, Re, Xe, Gf=0.0, Bf=0.0):
    delta, omega, Eqp, psidp, Edp, psiqp = x

    Id, Iq, Vdterm, Vqterm, Sa, psi_ag = solve_network(
        delta, omega, Eqp, psidp, Edp, psiqp, p, Einf, Re, Xe, Gf=Gf, Bf=Bf)

    Eqpp = Eqp * (p.Xdpp - p.Xl) / (p.Xdp - p.Xl) + psidp * (p.Xdp - p.Xdpp) / (p.Xdp - p.Xl)
    Edpp = Edp * (p.Xqpp - p.Xl) / (p.Xqp - p.Xl) + psiqp * (p.Xqp - p.Xqpp) / (p.Xqp - p.Xl)

    Xdpp_sat = p.Xl + (p.Xdpp - p.Xl) / (1 + Sa)
    Xqpp_sat = p.Xl + (p.Xqpp - p.Xl) / (1 + Sa)

    psi_d = Eqpp - Id * Xdpp_sat
    psi_q = -Edpp - Iq * Xqpp_sat
    Telec = psi_d * Iq - psi_q * Id

    ddelta = omega * p.omega0
    domega = (1.0 / (2 * p.H)) * ((Pmech - p.D * omega) / (1 + omega) - Telec)

    LadIfd = field_current(Eqp, psidp, Id, Sa, p)

    dEqp = (Efd - LadIfd) / p.Tdop
    dpsidp = ((1 + Sa) * (-psidp - (p.Xdp - p.Xl) * Id / (1 + Sa) + Eqp)) / p.Tdopp
    dpsiqp = ((1 + Sa) * (-psiqp + (p.Xqp - p.Xl) * Iq / (1 + Sa) + Edp)) / p.Tqopp
    dEdp = ((1 + Sa) * (-Edp + (p.Xq - p.Xqp) * Iq / (1 + Sa)
             - (p.Xqp - p.Xqpp) / (p.Xqp - p.Xl) ** 2
             * (Edp - psiqp + (p.Xqp - p.Xl) * Iq / (1 + Sa)))) / p.Tqop

    return np.array([ddelta, domega, dEqp, dpsidp, dEdp, dpsiqp])


def recover_algebraic(x_arr, p, Einf, Re, Xe, Gf=0.0, Bf=0.0):
    n = x_arr.shape[1]
    Gf_arr = np.broadcast_to(np.asarray(Gf, dtype=float), (n,))
    Bf_arr = np.broadcast_to(np.asarray(Bf, dtype=float), (n,))
    out = {k: np.zeros(n) for k in
           ["Id", "Iq", "Vdterm", "Vqterm", "Vt", "P", "Q", "Sa", "psi_ag"]}
    sa = 1.0
    for i in range(n):
        delta, omega, Eqp, psidp, Edp, psiqp = x_arr[:, i]
        Id, Iq, Vd, Vq, sa, psi = solve_network(
            delta, omega, Eqp, psidp, Edp, psiqp, p, Einf, Re, Xe,
            Gf=Gf_arr[i], Bf=Bf_arr[i], sat_guess=sa)
        out["Id"][i], out["Iq"][i] = Id, Iq
        out["Vdterm"][i], out["Vqterm"][i] = Vd, Vq
        out["Vt"][i] = np.hypot(Vd, Vq)
        out["P"][i] = Vd * Id + Vq * Iq
        out["Q"][i] = Vq * Id - Vd * Iq
        out["Sa"][i], out["psi_ag"][i] = sa, psi
    return out


# ---------------------------------------------------------------------
# Closed-form steady-state initialisation (Sec. 1.1 / 1.1.1). No fsolve
# is needed: psi_ag can be evaluated in the network reference frame
# before delta is known (rotation-invariance), so Sa, then delta (closed
# form, Sec 1.1.1), then every state, follow sequentially.
# ---------------------------------------------------------------------

def initialize(p: GENQECParams, Vt0, P0, Q0, Re, Xe, Gf=0.0, Bf=0.0):
    """Returns (state0, Efd0, Pmech0, Einf_mag). Gf,Bf: shunt admittance
    at the machine terminal (see solve_network docstring) -- a permanent
    local load, in this project's case. Only the Einf back-calculation
    changes: the machine's OWN (Id0,Iq0,Vd0,Vq0,...) still come directly
    from (P0,Q0) exactly as before (that IS the machine's own terminal
    output, unaffected by what happens to it afterwards) -- what changes
    is that only part of It0 continues on through Re+jXe to the infinite
    bus, the rest being drawn off locally by the shunt at the SAME bus.
    """
    It0 = complex(P0, -Q0) / Vt0        # network frame, Vt0 = Vt0<0deg
    Vr0, Vi0 = Vt0, 0.0
    Ir0, Ii0 = It0.real, It0.imag

    # psi_ag from network-frame quantities directly (no delta needed)
    Vrr0 = Vr0 + Ir0 * p.Ra - Ii0 * p.Xl
    Vir0 = Vi0 + Ii0 * p.Ra + Ir0 * p.Xl
    psi_ag0 = np.sqrt(Vrr0 ** 2 + Vir0 ** 2)     # omega0 = 0
    Sa0 = float(p.sat(psi_ag0))

    # closed-form rotor angle (Sec. 1.1.1)
    Xq_tilde = (p.Xq - p.Xl) / (1 + Sa0) + p.Xl
    delta_rel = np.arctan2(Vi0 + p.Ra * Ii0 + Ir0 * Xq_tilde,
                            Vr0 + p.Ra * Ir0 - Ii0 * Xq_tilde)

    rot = np.exp(-1j * (delta_rel - np.pi / 2))
    Id0, Iq0 = (It0 * rot).real, (It0 * rot).imag
    Vd0, Vq0 = (complex(Vt0, 0.0) * rot).real, (complex(Vt0, 0.0) * rot).imag  # Vdterm0, Vqterm0

    Xdpp_sat0 = p.Xl + (p.Xdpp - p.Xl) / (1 + Sa0)
    Xqpp_sat0 = p.Xl + (p.Xqpp - p.Xl) / (1 + Sa0)

    Eqpp0 = Vq0 + Xdpp_sat0 * Id0 + p.Ra * Iq0                    # = psi_d''0
    psidp0 = Eqpp0 - (p.Xdpp - p.Xl) * Id0 / (1 + Sa0)            # closed form (verified)
    Eqp0 = psidp0 + (p.Xdp - p.Xl) * Id0 / (1 + Sa0)              # from eq.1

    Edp0 = (p.Xq - p.Xqp) * Iq0 / (1 + Sa0)                       # eq.3
    psiqp0 = Edp0 + (p.Xqp - p.Xl) * Iq0 / (1 + Sa0)              # eq.4

    Edpp0 = Vd0 + p.Ra * Id0 - Xqpp_sat0 * Iq0                    # cross-check target = -psi_q''0

    Kidw0 = np.clip(p.Kw * Id0, -0.25, 0.25)
    LadIfd0 = (1 + Sa0) / (1 - Kidw0) * (Eqp0 + (p.Xd - p.Xdp) * Id0 / (1 + Sa0))
    Efd0 = LadIfd0

    psi_d0 = Eqpp0 - Id0 * Xdpp_sat0
    psi_q0 = -Edpp0 - Iq0 * Xqpp_sat0
    Telec0 = psi_d0 * Iq0 - psi_q0 * Id0
    Pmech0 = Telec0

    Y_shunt = complex(Gf, Bf)
    I_shunt0 = Y_shunt * complex(Vt0, 0.0)     # local draw at the machine bus, network frame
    I_to_inf0 = It0 - I_shunt0                 # only this part actually crosses Re+jXe
    Einf_phasor = complex(Vt0, 0.0) - I_to_inf0 * (Re + 1j * Xe)
    Einf_mag = abs(Einf_phasor)
    theta_t0 = -np.angle(Einf_phasor)
    delta0 = delta_rel + theta_t0

    P_check = Vd0 * Id0 + Vq0 * Iq0
    Q_check = Vq0 * Id0 - Vd0 * Iq0
    if abs(P_check - P0) > 1e-7 or abs(Q_check - Q0) > 1e-7:
        raise RuntimeError(f"Init power mismatch: P_check={P_check}, Q_check={Q_check}")

    state0 = np.array([delta0, 0.0, Eqp0, psidp0, Edp0, psiqp0])
    return state0, Efd0, Pmech0, Einf_mag
