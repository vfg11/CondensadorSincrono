"""
plc_export.py
================
Genera un documento de referencia, generico (no atado a ninguna marca
de PLC), con las matrices EXACTAS del observador+controlador LQI ya
disenado en el estudio actual, y la secuencia de calculo por ciclo --
para que el usuario lo traduzca a su plataforma concreta (ladder, texto
estructurado, C, etc).
"""
import numpy as np
from scipy.linalg import expm
from datetime import datetime


def compute_discrete_observer(Ar, Cr, L, Br, Ts):
    """Discretizacion EXACTA (Van Loan) del observador, descompuesta en
    forma discreta estandar:

        x_hat[k+1] = Ad @ x_hat[k] + Bd_u * u_prev[k] + Bd_y * y_meas[k]

    en vez de dejar la exponencial de matriz para calcular en el PLC.
    Mismo metodo que usan plc_battery.py y linear_test.py -- verificado
    aqui mismo que da resultados identicos antes de exportarlo."""
    n = Ar.shape[0]
    Aobs = Ar - L.flatten()[:, None] @ Cr
    Ad = expm(Aobs * Ts)

    Mu = np.zeros((n + 1, n + 1))
    Mu[:n, :n] = Aobs
    Mu[:n, n] = Br.flatten()
    Bd_u = expm(Mu * Ts)[:n, n]

    My = np.zeros((n + 1, n + 1))
    My[:n, :n] = Aobs
    My[:n, n] = L.flatten()
    Bd_y = expm(My * Ts)[:n, n]

    return Ad, Bd_u, Bd_y


def _format_matrix(M, name, decimals=8):
    M = np.atleast_2d(M)
    lines = [f"{name} ({M.shape[0]}x{M.shape[1]}):"]
    for row in M:
        vals = ", ".join(f"{v:.{decimals}g}" for v in row)
        lines.append(f"  [{vals}]")
    return "\n".join(lines)


def _format_vector(v, name, decimals=8):
    v = np.atleast_1d(v).flatten()
    vals = ", ".join(f"{x:.{decimals}g}" for x in v)
    return f"{name} ({len(v)}):\n  [{vals}]"


def build_plc_export(params, objs, lin_result, red_result, design_result):
    """Devuelve el documento completo como string de texto."""
    Ar, Br, Cr = red_result['Ar'], red_result['Br'], red_result['Cr']
    K, L = design_result['K'], design_result['L']
    Ts = design_result['Ts']
    n = Ar.shape[0]
    Kx = K[0, :n]
    Ki = K[0, n]
    Vref0 = lin_result['Vref0']
    AVR = objs['AVR']
    Ad, Bd_u, Bd_y = compute_discrete_observer(Ar, Cr, L, Br, Ts)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = []
    doc.append("=" * 78)
    doc.append("IMPLEMENTACION DE LOS REGULADORES LQI Y PI EN UN PLC -- REFERENCIA GENERICA")
    doc.append(f"Generado: {now}")
    doc.append("=" * 78)
    doc.append("")
    doc.append("Este documento NO es codigo de ningun PLC concreto -- es la formulacion")
    doc.append("matematica exacta y la secuencia de calculo por ciclo de AMBOS reguladores")
    doc.append("usados en la comparativa de este estudio, para traducir a la plataforma")
    doc.append("que se use (texto estructurado, ladder+funciones, C, etc). Son DOS")
    doc.append("reguladores INDEPENDIENTES -- implementar uno u otro, no los dos a la vez.")
    doc.append("")

    doc.append("-" * 78)
    doc.append("1. CONSTANTES DEL PUNTO DE EQUILIBRIO (calcular una vez, offline)")
    doc.append("-" * 78)
    doc.append(f"Ts (ciclo de ejecucion)      = {Ts:.6g} s")
    doc.append(f"Vref0 (Vref de equilibrio)   = {Vref0:.8g} pu")
    doc.append(f"VREF_MIN                     = {AVR.VREF_MIN:.6g} pu")
    doc.append(f"VREF_MAX                     = {AVR.VREF_MAX:.6g} pu")
    doc.append(f"Qeq (Q de equilibrio, referencia de desviacion) = "
               f"{'<< calcular en el punto de arranque real, ver seccion 5 >>'}")
    doc.append(f"Orden del observador (n)     = {n}")
    doc.append("")

    doc.append("-" * 78)
    doc.append("2. MATRICES DEL LQI (calcular/cargar una vez, offline -- constantes)")
    doc.append("-" * 78)
    doc.append(_format_matrix(Ar, "Ar (dinamica de la planta reducida)"))
    doc.append("")
    doc.append(_format_matrix(Br, "Br (entrada de la planta reducida)"))
    doc.append("")
    doc.append(_format_matrix(Cr, "Cr (salida medida, Qdeliv_m)"))
    doc.append("")
    doc.append(_format_vector(L, "L (ganancia del observador)"))
    doc.append("")
    doc.append(_format_vector(Kx, "Kx (ganancia de realimentacion de estado)"))
    doc.append(f"Ki (ganancia integral)       = {Ki:.8g}")
    doc.append("")
    doc.append("Nota: Kx y Ki salen de UNA sola matriz K (1x(n+1)) disenada sobre el")
    doc.append("sistema aumentado [Ar 0; -Cr 0] -- no se disenan por separado.")
    doc.append("")

    doc.append("-" * 78)
    doc.append("3. ESTADO PERSISTENTE DEL LQI ENTRE CICLOS (variables, no constantes)")
    doc.append("-" * 78)
    doc.append(f"x_hat[0..{n-1}]   : estado estimado del observador. Inicializar a 0.")
    doc.append("xi               : estado integral. Inicializar a 0.")
    doc.append("Vref_aplicado    : ultimo Vref enviado al AVR. Inicializar a Vref0.")
    doc.append("")

    doc.append("-" * 78)
    doc.append("4. MATRICES DISCRETAS DEL OBSERVADOR, YA CALCULADAS PARA Ts DE ESTE ESTUDIO")
    doc.append("-" * 78)
    doc.append("El observador continuo es:")
    doc.append("  dx_hat/dt = Ar@x_hat + Br*u_prev + L*(y_meas - Cr@x_hat)")
    doc.append("           = (Ar - L@Cr)@x_hat + Br*u_prev + L*y_meas")
    doc.append("")
    doc.append(f"Discretizado EXACTAMENTE (exponencial de matriz, metodo de Van Loan) para")
    doc.append(f"Ts={Ts:.6g}s -- NO es una aproximacion, y NO hace falta calcular ninguna")
    doc.append(f"exponencial en el PLC: son las 3 matrices constantes de abajo, cargadas")
    doc.append(f"una vez, con una unica actualizacion lineal por ciclo:")
    doc.append("")
    doc.append("  x_hat[siguiente] = Ad @ x_hat + Bd_u * u_prev + Bd_y * y_meas")
    doc.append("")
    doc.append(_format_matrix(Ad, "Ad (transicion de estado discreta)"))
    doc.append("")
    doc.append(_format_vector(Bd_u, "Bd_u (entrada discreta asociada a u_prev)"))
    doc.append("")
    doc.append(_format_vector(Bd_y, "Bd_y (entrada discreta asociada a y_meas)"))
    doc.append("")
    doc.append("Verificado en este estudio: un unico paso de Euler a Ts completo (en vez")
    doc.append("de esta discretizacion exacta) resulto NUMERICAMENTE INESTABLE para los")
    doc.append("polos internos rapidos de esta planta -- usar Ad/Bd_u/Bd_y de arriba, no")
    doc.append("Euler, salvo que se verifique especificamente que Euler no diverge para")
    doc.append("los valores reales de esta maquina.")
    doc.append("")
    doc.append("IMPORTANTE -- si se cambian Ar, Cr, L o Ts (por ejemplo, tras un nuevo")
    doc.append("diseno con parametros distintos), estas 3 matrices ESTAN LIGADAS a esos")
    doc.append("valores concretos y hay que recalcularlas -- no son universales.")
    doc.append("")

    doc.append("-" * 78)
    doc.append("5. VALIDACION DEL OBSERVADOR EN ABIERTO (recomendado antes de cerrar el lazo)")
    doc.append("-" * 78)
    doc.append("""
Antes de fiarse del LQI para regular, conviene comprobar que el modelo
reducido + observador capturan bien el sistema real -- es decir, que
Q_estimado por el observador sigue de cerca a Q_medida, ANTES de usar
x_hat para calcular ningun Vref nuevo.

Durante esta fase, Vref sigue su curso NORMAL (manual, u otro regulador
ya en marcha) -- el observador solo MIRA, no actua. Se necesitan
ADEMAS de Q_medida, lecturas de Vref real aplicado en cada ciclo.

PASO 1 -- Leer sensores:
    y_medida     = Q_entregada medida
    Vref_actual  = Vref realmente aplicado en este ciclo (venga de
                   donde venga -- manual, otro regulador, etc)

PASO 2 -- Calcular entrada/salida en desviacion (igual que en cierre de lazo):
    u_prev  = Vref_actual - Vref0
    y_meas  = y_medida - Qeq

PASO 3 -- Actualizar el observador (identico al cierre de lazo):
    x_hat = Ad @ x_hat + Bd_u * u_prev + Bd_y * y_meas

PASO 4 -- Calcular Q ESTIMADO (esto es lo nuevo de esta fase):
    Q_estimado = Qeq + Cr . x_hat

PASO 5 -- Comparar y registrar, NO actuar todavia:
    error_observador = y_medida - Q_estimado
    (registrar y_medida, Q_estimado, error_observador cada ciclo para
    su analisis -- no se calcula Vref nuevo en esta fase)
""")
    doc.append("Nota sobre la comparacion: Q_estimado del PASO 4 usa el x_hat ya")
    doc.append("actualizado con la MISMA medida y_medida de este ciclo (PASO 3) -- es")
    doc.append("una comprobacion de CONSISTENCIA del observador, no de prediccion pura.")
    doc.append("Para una prueba mas exigente (prediccion a un paso), comparar Q_estimado")
    doc.append("de este ciclo con y_medida del ciclo SIGUIENTE, antes de que esa medida")
    doc.append("entre en el PASO 3 de ese ciclo siguiente.")
    doc.append("")
    doc.append("Este estudio no fija un umbral de aceptacion universal para")
    doc.append("error_observador -- depende de la precision que la aplicacion necesite.")
    doc.append("Como referencia, en la bateria de tests de este estudio (ver resultados")
    doc.append("del paso 5 de la aplicacion) se puede inspeccionar la magnitud tipica de")
    doc.append("las desviaciones de Q durante transitorios, para tener una idea de la")
    doc.append("escala esperada.")
    doc.append("")

    doc.append("-" * 78)
    doc.append("6. SECUENCIA DE CALCULO DEL LQI EN CIERRE DE LAZO, CADA CICLO Ts")
    doc.append("-" * 78)
    doc.append("""
PASO 1 -- Leer sensor:
    y_medida = Q_entregada medida (misma señal/retardo que Qdeliv_m del estudio)

PASO 2 -- Calcular entrada/salida en desviacion:
    u_prev  = Vref_aplicado - Vref0
    y_meas  = y_medida - Qeq            (Qeq: ver nota abajo)

PASO 3 -- Actualizar el observador (matrices Ad, Bd_u, Bd_y de la seccion 4):
    x_hat = Ad @ x_hat + Bd_u * u_prev + Bd_y * y_meas

PASO 4 -- Actualizar el estado integral:
    xi = xi + (Q_referencia - y_medida) * Ts

PASO 5 -- Ley de control:
    Vref_sin_saturar = Vref0 - (Kx . x_hat) - Ki * xi
    Vref_saturado    = clip(Vref_sin_saturar, VREF_MIN, VREF_MAX)

PASO 6 -- Anti-windup (retrocalculo) -- EL SIGNO Y EL FACTOR Ts IMPORTAN, ver nota:
    SI Ki != 0:
        xi = xi - Ts * (Vref_saturado - Vref_sin_saturar) / Ki

PASO 7 -- Aplicar y guardar:
    enviar Vref_saturado al AVR
    Vref_aplicado = Vref_saturado
""")

    doc.append("=" * 78)
    doc.append("REGULADOR ALTERNATIVO: PI (mucho mas simple, sin observador)")
    doc.append("=" * 78)
    doc.append("")
    doc.append("El PI usado en la comparativa de este estudio -- realimenta DIRECTAMENTE")
    doc.append("el error medido, sin estimar ningun estado interno de la planta. Usa las")
    doc.append("MISMAS constantes de equilibrio de la seccion 1 (Ts, Vref0, VREF_MIN/MAX,")
    doc.append("Qeq) -- no hace falta repetirlas.")
    doc.append("")
    doc.append("-" * 78)
    doc.append("P1. GANANCIAS DEL PI (calcular/cargar una vez, offline -- constantes)")
    doc.append("-" * 78)
    doc.append(f"Kp (ganancia proporcional)   = {design_result['Kp']:.8g}")
    doc.append(f"Ki (ganancia integral)       = {design_result['Ki']:.8g}")
    doc.append("")
    doc.append("-" * 78)
    doc.append("P2. ESTADO PERSISTENTE DEL PI ENTRE CICLOS (variables, no constantes)")
    doc.append("-" * 78)
    doc.append("xi_pi            : estado integral. Inicializar a 0.")
    doc.append("Vref_aplicado    : ultimo Vref enviado al AVR. Inicializar a Vref0.")
    doc.append("(Nota: xi_pi es independiente del xi del LQI -- son dos reguladores")
    doc.append(" separados, cada uno con su propio estado integral, no compartido.)")
    doc.append("")
    doc.append("-" * 78)
    doc.append("P3. SECUENCIA DE CALCULO DEL PI, CADA CICLO Ts")
    doc.append("-" * 78)
    doc.append("""
PASO 1 -- Leer sensor:
    y_medida = Q_entregada medida

PASO 2 -- Calcular error:
    error = Q_referencia - y_medida

PASO 3 -- Actualizar el estado integral:
    xi_pi = xi_pi + error * Ts

PASO 4 -- Ley de control:
    Vref_sin_saturar = Vref0 + Kp * error + Ki * xi_pi
    Vref_saturado    = clip(Vref_sin_saturar, VREF_MIN, VREF_MAX)

PASO 5 -- Anti-windup (retrocalculo) -- SIGNO OPUESTO al del LQI, ver nota:
    SI Ki != 0:
        xi_pi = xi_pi + Ts * (Vref_saturado - Vref_sin_saturar) / Ki

PASO 6 -- Aplicar y guardar:
    enviar Vref_saturado al AVR
    Vref_aplicado = Vref_saturado
""")
    doc.append("Sin PASO de observador (a diferencia del LQI) -- no hace falta la seccion")
    doc.append("4 de discretizacion, el PI no estima ningun estado interno.")
    doc.append("")

    doc.append("-" * 78)
    doc.append("7. NOTAS IMPORTANTES, VERIFICADAS EN ESTE PROYECTO (aplican a ambos)")
    doc.append("-" * 78)
    doc.append("""
- Qeq (LQI paso 2, PI paso 2): equilibrio REAL del punto donde arranca
  el regulador, NO un valor fijo de tabla. Medirlo/calcularlo en el
  arranque (con la planta en reposo, Qeq = Q medida en ese instante),
  no usar un valor precalculado de otro punto de operacion -- un
  desajuste aqui genera un transitorio de arranque espurio (verificado,
  ver conversacion).

- Anti-windup (LQI paso 6, PI paso 5): esta formula se corrigio DOS
  VECES en este proyecto tras encontrar sendos errores -- uno afectaba
  a AMBOS reguladores (faltaba el factor Ts, la correccion resultaba
  ~10 veces mas fuerte de lo debido y podia generar un "punto fijo"
  espurio: el regulador se quedaba pegado a un limite aunque el error
  real ya hubiera cambiado de signo); el segundo afectaba SOLO al LQI
  (su ley de control tiene un signo menos explicito delante de
  Ki*xi -- Vref = Vref0 - Kx.x_hat - Ki*xi, a diferencia del PI que
  tiene signo mas -- Vref = Vref0 + Kp*e + Ki*xi -- asi que el LQI
  necesita el signo OPUESTO al del PI en esta formula, no el mismo).
  NINGUNO de los dos errores se notaba con una saturacion leve o un
  escalon moderado -- solo se revelaban con una saturacion SEVERA Y
  SOSTENIDA, donde el estado integral divergia sin limite (verificado:
  en un caso concreto crecio a mas de mil millones en 30 segundos)
  mientras la señal de salida parecia comportarse con normalidad,
  enmascarando el problema. Si se traduce esta logica a otra
  plataforma o se modifica el diseno, PROBARLA siempre con un ensayo
  de saturacion severa antes de darla por buena -- un escalon suave no
  es suficiente para confiar en que el anti-windup esta bien
  implementado.

- El signo/factor correctos, en TERMINOS GENERALES: identificar como
  aparece Ki multiplicando al estado integral en la ley de control
  propia (con signo mas o signo menos), y elegir el signo de esta
  formula de anti-windup de forma que, durante saturacion SEVERA y
  SOSTENIDA, el estado integral quede ACOTADO (no crezca sin limite).
  Probar ambos signos si hay duda -- el signo equivocado diverge de
  forma inequivoca bajo esa prueba, no hay ambiguedad en el resultado.

- Vref_aplicado es lo que alimenta el ciclo SIGUIENTE (u_prev del LQI,
  o simplemente el punto de partida del PI) -- no Vref_sin_saturar. El
  observador del LQI en particular debe ver lo que REALMENTE se aplico,
  no lo que el regulador hubiera querido aplicar.

- El AVR (con su propio lazo interno PI+techo de excitacion) es una
  pieza APARTE, ya implementada en el equipo real -- Vref es la SALIDA
  de estos reguladores y la ENTRADA de ese AVR, no hay que
  reimplementar el AVR.

- El PI es notablemente mas simple de implementar (sin matrices, sin
  observador, dos constantes) pero en la comparativa de este estudio
  resulto sistematicamente mas lento que el LQI -- ver los resultados
  de la bateria de tests para la magnitud concreta de esa diferencia
  en este caso.
""")

    return "\n".join(doc)
