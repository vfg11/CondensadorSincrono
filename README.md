# Condensador síncrono, máquina nueva — estado actual

## 1. Qué hay aquí y qué falta

Pipeline hasta ahora: **derivación simbólica → linealización → reducción
de orden → PI diseñado y verificado (no lineal)**. Falta: **LQI**
(siguiente paso, no empezado) y el resto de la batería del punto 6
(escalones fuera del punto de diseño, perturbación de tensión de red) —
esos aplican mejor una vez estén LQI y PI listos para comparar juntos.

M�quina: H=15, D=0 reales; resto (Xd, Xq, Tdo', Tqo'...) sigue en el
juego **ilustrativo** de siempre (Xd=1.80 etc.) pendiente de datos
reales. AVR, carga (8%, cosφ=0.85), límite Vref≤1.15 y retardos de
0.08s en el punto de entrega: todo real.

## 2-3. `01_model/`, `02_linearization/` — AVR + carga + retardos (11 estados)

Sin cambios desde la sesión anterior. AVR de 3 estados (PI + techo FEX,
T_MEAS=0.0111s en Vt e Ifd por separado), carga como admitancia shunt
en el terminal (reproduce "P de la red, Q del condensador" de forma
exacta), Vref limitado a 1.15, dos lags de 0.08s en Vdelivery/Q_delivered.
Verificado por diferencias finitas, PASS (~1e-7).

## 4. `03_design/reduce_and_design_pi.py` — reducción + PI (primera pasada)

**Reducción**: Hankel con salida única Qdeliv_m (confirmaste que el lazo
regula reactiva). Salto claro 6→7 (factor ~128x). `Vdeliv_m` resulta
EXACTAMENTE inobservable desde Q sola (σ11=0.0 exacto — no influye en
nada más y Q no depende de ella, no es un redondeo). matchdc a orden 6,
error frente al modelo completo: 5.6e-5 en escalón en lazo abierto (3s).

**PI, primera pasada (solo modelo lineal)**: rejilla en (Kp,Ki)
minimizando tiempo de establecimiento (banda 2%) — igual que hacía
`run_pi_tuning()` en el proyecto anterior. Resultado: Kp=0.4124,
Ki=1.6388, con predicción de 0.336s / 1.89% de sobreimpulso.

## 5. `04_simulation/validate_pi.py` — de "óptimo lineal" a PI real

**Hallazgo clave**: ese PI "óptimo" probado contra el modelo NO lineal
completo da 2.015s / 33.6% de sobreimpulso — muy lejos de lo previsto.
Causa: el PI *interno* del AVR (Kp=72, U_MAX=3.25) solo tiene ~0.045pu
de margen en el error de Vt antes de saturar. Casi cualquier corrección
del lazo externo lo satura, sobre todo justo tras un escalón (antes de
que reaccione el lag de Vt_m) — el modelo lineal no lo ve porque asume
que nunca se sale de zona no saturada. Se explica solo, y probablemente
se repetirá con LQI si no se tiene en cuenta.

**PI final**, buscado directamente contra el modelo no lineal (esta vez
sí, la búsqueda "ve" la saturación): **Kp=0.06, Ki=0.6**.

- Escalón +0.15 pu en Q_ref: **1.08s** de establecimiento, **1.6%** de
  sobreimpulso, Vref llega solo a 1.043 (ni roza el límite de 1.15).
- Cortocircuito trifásico 150ms: Vref satura correctamente en 1.15
  durante la falla, el estado integral sube a 0.38 y vuelve a ~0
  (0.00017) sin divergir, Q se recupera con error 8.6e-5 a los 12s.
  **Antiwindup verificado** con el ensayo de saturación severa, tal
  como recomienda el resumen.

Antiwindup implementado por retrocálculo continuo estándar (no la
fórmula discreta exacta del resumen — su signo no se puede reconstruir
con fiabilidad sin el código original — pero verificado igual de a
fondo: el ensayo de falla es justo el que expone ese tipo de bug).

Nota técnica: la búsqueda evitó Ki=0.7 — esa franja resulta
numéricamente patológica para el integrador (Radau se atasca) en varios
puntos, independientemente de Kp; no aporta nada que Ki=0.6 no cubra.

Gráficas en `outputs/validate_pi.png`.

## 6. Para la próxima sesión

LQI: mismo orden reducido (6, salida Qdeliv_m), observador de Kalman,
integral aumentada, y — visto lo del PI — conviene verificar desde el
principio contra el modelo no lineal, no fiarse del diseño puramente
lineal. Después: escalones fuera del punto de diseño y perturbación de
tensión de red, con LQI y PI ya comparables.

## 7. `03_design/design_lqi.py` + `04_simulation/validate_lqi.py` — LQI

Mismo orden reducido (6, salida Qdeliv_m). Estado aumentado con
integral z=[x(6), xi(1)]; LQR (`solve_continuous_are`) con pesos
qy*(salida)² + qi*xi² + eps·||x||² + r·u² (mismo espíritu que el
Qy/peso-integral/R del proyecto anterior, adaptado a una sola salida).
Observador de Kalman (ruido de proceso entrando por el canal de
entrada, ruido de medida en Qdeliv_m) colocado ~4x más rápido que el
lazo cerrado del controlador.

**Nota de diseño**: el polo más lento del modelo reducido (−0.464)
resultó prácticamente fijo — no se mueve ni con pesos extremos
(probado hasta qi=1e7). Es débilmente controlable Y débilmente
observable desde Q (proyección ~6.5% en ambos casos, ni cero ni
dominante) — el LQR, correctamente, no gasta esfuerzo en moverlo
porque apenas influye en el coste de seguimiento. No es un fallo del
diseño, es una propiedad estructural del par (Ar,Br,Cr) con esta
salida.

**Mismo hallazgo que con el PI** (y van dos, así que probablemente es
sistemático con este AVR): los pesos "óptimos" del barrido lineal
(qy=2.848, qi=1000, predicción rápida y limpia) saturan el AVR interno
de inmediato contra el modelo no lineal — 1.327s / 47.3% de
sobreimpulso, peor incluso que el PI equivalente. Refinado directamente
contra el modelo no lineal: **qy=1.4, qi=20**.

### Comparativa final (escalón +0.15 pu en Q_ref, mismas condiciones)

| | t_settle (2%) | sobreimpulso | Vref máx |
|---|---|---|---|
| PI (Kp=0.06, Ki=0.6) | 1.080 s | 1.61 % | 1.043 |
| LQI (qy=1.4, qi=20) | **0.712 s** | 1.39 % | 1.077 |

LQI ~34% más rápido que el PI con sobreimpulso comparable, a costa de
acercarse algo más al límite de Vref (1.077 vs 1.043, ambos con margen
razonable frente a 1.15).

**Cortocircuito trifásico 150ms**: Vref satura correctamente en 1.15,
integral vuelve a ~0 (1.75e-5) sin divergir, Q se recupera con error
8.06e-6 a los 12s — antiwindup verificado igual que con el PI.

Gráficas en `outputs/validate_lqi.png`.

## 8. Para la próxima sesión

Quedan de la batería del punto 6 del resumen: escalones fuera del
punto de diseño (Q0 distinto de 0.35) y perturbación de tensión de red
(Einf) — con LQI y PI ya listos y comparables. Ojo con el punto de
equilibrio real vs el de diseño al cambiar de Q0 (ver sec.5 del
resumen) — `init_plant()`/`_init_z0()` ya usan el equilibrio real, no
uno fijo, así que debería estar cubierto, pero conviene verificarlo
explícitamente la primera vez que se pruebe fuera del punto de diseño.

## 9. `04_simulation/battery_offdesign_and_grid.py` — puntos 2 y 4 de la batería

Mismas ganancias fijas (PI: Kp=0.06,Ki=0.6; LQI: qy=1.4,qi=20) que en
el punto de diseño — sin rediseñar. Esta es la prueba real de si un
único punto de diseño se generaliza, la premisa de todo el enfoque
single-point (frente a gain-scheduling/LPV).

### Punto 2: escalón +0.15 pu fuera del punto de diseño (Q0=0.35)

| Q0 | PI t_settle | PI sobreimpulso | LQI t_settle | LQI sobreimpulso | Vref máx (PI / LQI) |
|---|---|---|---|---|---|
| 0.00 | 1.01 s | 1.61% | 1.15 s | ~0% | 1.040 / 1.039 |
| 0.60 | 4.00 s | 0% | 3.67 s | 1.23% | 1.105 / **1.150 (satura)** |

Generaliza razonablemente bien en Q0=0.0 (parecido al punto de diseño).
En Q0=0.60 (cerca del borde del rango alcanzable — recuerda que
Q0=0.85 ya ni siquiera tiene equilibrio válido para este AVR) ambos se
vuelven ~3.5-4x más lentos, y el LQI llega a saturar Vref exactamente
en el límite. Ninguno se vuelve inestable, pero el rendimiento se
degrada notablemente al alejarse del punto de diseño — esperable, y
es justo la clase de límite que este ensayo está pensado para
encontrar.

### Punto 4: perturbación de tensión de red (Einf), Q_ref fija

| Escalón Einf | PI desv.máx Q | PI error final | LQI desv.máx Q | LQI error final |
|---|---|---|---|---|
| −10% | 0.154 | 6.1e-4 | 0.062 | 5.7e-5 |
| −5% | 0.088 | 2.1e-4 | 0.031 | 3.2e-5 |
| +5% | 0.096 | 1.5e-4 | 0.040 | 1.2e-5 |
| +10% | 0.201 | 2.9e-4 | 0.099 | **1.9e-6** (Vref satura en 1.15) |

El LQI rechaza la perturbación de red sistemáticamente mejor que el PI
en los cuatro casos — desviación máxima ~2.5x menor, error final
~10x-100x menor — consistente con tener realimentación de estado
completo en vez de un único lazo P+I. El PI nunca satura Vref en este
ensayo (máximo 1.085 en +10%); el LQI sí, justo en el caso más extremo
(+10%), aunque aun así termina con el error final más bajo de toda la
tabla.

Gráficas en `outputs/battery_offdesign_and_grid.png`.

## 10. Batería completa — resumen

Los 4 puntos de la sección 8 del resumen están cubiertos para ambos
controladores: escalón en el punto de diseño, escalón fuera de diseño,
rechazo de falla (cortocircuito trifásico 150ms), perturbación de
tensión de red. En los cinco ensayos que admiten comparación directa,
el LQI iguala o supera al PI — más rápido en el escalón de diseño
(0.71s vs 1.08s), mejor rechazo de perturbación de red en los cuatro
casos — a cambio de acercarse algo más al límite de Vref en las
condiciones más exigentes (Q0=0.6, Einf+10%). El PI, más simple, se
mantiene siempre con más margen frente al límite.

Pendiente real: sigue sin haber datos reales de la máquina (solo H=15,
D=0 lo son) — todo lo anterior usa el juego ilustrativo de Xd/Xq/Tdo'
de siempre. Sería el paso lógico para dar por definitivamente cerrado
este estudio.
