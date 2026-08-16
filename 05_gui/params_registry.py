"""
params_registry.py
=====================
Registro centralizado de TODOS los parametros ajustables (maquina, AVR,
red/carga, punto de operacion). Una unica fuente de verdad: de aqui
sale tanto el dialogo con pestanas (cada campo se dibuja solo) como el
formato de guardado/carga de proyecto.

Cada parametro es una tupla (nombre, etiqueta, valor_por_defecto,
minimo, maximo, decimales) agrupada por pestana. Todos los campos
admiten hasta 8 decimales de precision. H no tiene limite superior
practico (se usa un maximo muy grande en vez de infinito, que
QDoubleSpinBox no admite).
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

import genqec_model as gqc
import controls as ctrl

_DEC = 8
_NO_LIMIT = 1.0e12   # "sin limite practico" -- QDoubleSpinBox exige un maximo finito

# (nombre, etiqueta, valor_por_defecto, minimo, maximo, decimales)
MACHINE_FIELDS = [
    ('S_nominal', 'S nominal (potencia aparente del condensador) [MVA]', 330.0, 0.001, _NO_LIMIT, _DEC),
    ('Ra', 'Ra (resistencia armadura) [pu]', 0.003, 0.0, 0.1, _DEC),
    ('Xl', 'Xl (dispersion) [pu]', 0.15, 0.0, 1.0, _DEC),
    ('Xd', 'Xd [pu]', 1.80, 0.1, 5.0, _DEC),
    ('Xdp', "Xd' [pu]", 0.30, 0.01, 3.0, _DEC),
    ('Xdpp', "Xd'' [pu]", 0.22, 0.01, 2.0, _DEC),
    ('Xq', 'Xq [pu]', 1.70, 0.1, 5.0, _DEC),
    ('Xqp', "Xq' [pu]", 0.50, 0.01, 3.0, _DEC),
    ('Xqpp', "Xq'' [pu]", 0.25, 0.01, 2.0, _DEC),
    ('Tdop', "Tdo' [s]", 7.5, 0.1, 20.0, _DEC),
    ('Tdopp', "Tdo'' [s]", 0.03, 0.001, 1.0, _DEC),
    ('Tqop', "Tqo' [s]", 0.50, 0.01, 5.0, _DEC),
    ('Tqopp', "Tqo'' [s]", 0.05, 0.001, 1.0, _DEC),
    ('H', 'H (inercia) [MWs/MVA]', 15.0, 0.001, _NO_LIMIT, _DEC),
    ('D', 'D (amortiguamiento)', 0.0, 0.0, _NO_LIMIT, _DEC),
    ('Kw', 'Kw (compensacion saturacion campo)', 0.20, 0.0, 2.0, _DEC),
    ('f0', 'f0 (frecuencia nominal) [Hz]', 60.0, 40.0, 70.0, _DEC),
]
SAT_FIELDS = [
    ('sat_a', 'Saturacion: parametro A', 0.10, 0.0, 2.0, _DEC),
    ('sat_b', 'Saturacion: parametro B', 0.30, 0.0, 2.0, _DEC),
]

AVR_FIELDS = [
    ('Kp', 'Kp (PI interno del AVR)', 72.0, 0.0, 500.0, _DEC),
    ('Ki', 'Ki (PI interno del AVR)', 35.0, 0.01, 500.0, _DEC),
    ('U_MIN', 'U_MIN (limite inferior salida PI interno del AVR)', -2.6, -20.0, 0.0, _DEC),
    ('U_MAX', 'U_MAX (limite superior salida PI interno del AVR)', 3.25, 0.0, 20.0, _DEC),
    ('KC', 'KC (constante de carga rectificador)', 0.0308, 0.0, 1.0, _DEC),
    ('FACTOR_MAX', 'FACTOR_MAX (techo de excitacion)', 1.4, 0.5, 5.0, _DEC),
    ('T_MEAS', 'T_MEAS (retardo de medida Vt/Ifd) [s]', 0.0111, 0.0001, 1.0, _DEC),
    ('VREF_MIN', 'VREF_MIN (limite inferior salida del regulador LQI/PI) [pu]', 0.95, 0.5, 1.5, _DEC),
    ('VREF_MAX', 'VREF_MAX (limite superior salida del regulador LQI/PI) [pu]', 1.05, 0.5, 1.5, _DEC),
]

NETWORK_FIELDS = [
    ('Rt', 'Rt (transformador) [pu]', 0.002, 0.0, 0.5, _DEC),
    ('Xt', 'Xt (transformador) [pu]', 0.12, 0.0, 1.0, _DEC),
    ('Rline', 'Rline (linea) [pu]', 0.01, 0.0, 0.5, _DEC),
    ('Xline', 'Xline (linea) [pu]', 0.08, 0.0, 1.0, _DEC),
    ('Rgrid', 'Rgrid (resto de red) [pu]', 0.01, 0.0, 0.5, _DEC),
    ('Xgrid', 'Xgrid (resto de red) [pu]', 0.06, 0.0, 1.0, _DEC),
    ('S_LOAD', 'S_LOAD (carga local, S nominal) [pu]', 0.08, 0.0, 1.0, _DEC),
    ('COSPHI', 'cos(phi) de la carga local', 0.85, 0.0, 1.0, _DEC),
    ('TDELIV', 'TDELIV (retardo medida punto de entrega) [s]', 0.08, 0.001, 1.0, _DEC),
]

OPERATING_POINT_FIELDS = [
    ('Vt0', 'Vt0 (tension terminal objetivo) [pu]', 1.0, 0.8, 1.2, _DEC),
    ('P0_OP', 'P0 (potencia activa) [pu]', 0.0, -1.0, 1.0, _DEC),
    ('Q0_OP', 'Q0 (punto de diseno, potencia reactiva) [pu]', 0.35, -1.0, 1.0, _DEC),
]

DESIGN_FIELDS = [
    ('Ts', 'Ts (ciclo de ejecucion del PLC) [s]', 0.1, 0.001, 2.0, _DEC),
]

ALL_GROUPS = {
    'Maquina': MACHINE_FIELDS,
    'Saturacion': SAT_FIELDS,
    'AVR': AVR_FIELDS,
    'Red y carga': NETWORK_FIELDS,
    'Punto de operacion': OPERATING_POINT_FIELDS,
    'Diseno': DESIGN_FIELDS,
}


def default_params_dict():
    d = {}
    for fields in ALL_GROUPS.values():
        for name, label, default, lo, hi, dec in fields:
            d[name] = default
    return d


def build_objects(params):
    """S_nominal es un parametro de REFERENCIA/documentacion -- el
    pipeline electrico trabaja en por unidad y no lo necesita para
    calcular nada, pero se guarda con el estudio y se incluye en la
    exportacion PLC para dejar constancia de a que maquina corresponde
    el diseno."""
    sat = gqc.make_saturation('quadratic', params['sat_a'], params['sat_b'])
    P_REAL = gqc.GENQECParams(
        Ra=params['Ra'], Xl=params['Xl'], Xd=params['Xd'], Xdp=params['Xdp'],
        Xdpp=params['Xdpp'], Xq=params['Xq'], Xqp=params['Xqp'], Xqpp=params['Xqpp'],
        Tdop=params['Tdop'], Tdopp=params['Tdopp'], Tqop=params['Tqop'],
        Tqopp=params['Tqopp'], H=params['H'], D=params['D'], Kw=params['Kw'],
        sat=sat, f0=params['f0'])
    AVR = ctrl.AVRParams(
        Kp=params['Kp'], Ki=params['Ki'], U_MIN=params['U_MIN'], U_MAX=params['U_MAX'],
        KC=params['KC'], FACTOR_MAX=params['FACTOR_MAX'], T_MEAS=params['T_MEAS'],
        VREF_MAX=params['VREF_MAX'], VREF_MIN=params['VREF_MIN'])

    Re_total = params['Rt'] + params['Rline'] + params['Rgrid']
    Xe_total = params['Xt'] + params['Xline'] + params['Xgrid']
    R1 = params['Rt'] + params['Rline']
    X1 = params['Xt'] + params['Xline']

    P_LOAD = params['S_LOAD'] * params['COSPHI']
    Q_LOAD = params['S_LOAD'] * (1 - params['COSPHI'] ** 2) ** 0.5
    GLOAD = P_LOAD / 1.0 ** 2
    BLOAD = -Q_LOAD / 1.0 ** 2

    return dict(P=P_REAL, AVR=AVR, Re_total=Re_total, Xe_total=Xe_total,
                R1=R1, X1=X1, Vt0=params['Vt0'], P0_OP=params['P0_OP'],
                GLOAD=GLOAD, BLOAD=BLOAD, TDELIV=params['TDELIV'],
                Q0_OP=params['Q0_OP'], Ts=params['Ts'], S_nominal=params['S_nominal'])
