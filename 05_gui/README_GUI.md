# GUI de estudio del condensador sincrono (PySide6)

## Requisitos
```
pip install PySide6 control slycot numpy scipy matplotlib
```
`slycot` es necesario para `control.balred` (paso de reduccion).

## Ejecutar
```
cd 05_gui
python3 main_app.py
```

## Flujo de trabajo

1. **Parametros de la maquina...** — dialogo con pestanas (Maquina,
   Saturacion, AVR, Red y carga, Punto de operacion, Diseno). El
   `Diseno` incluye `Ts`, el ciclo de ejecucion del PLC.
2. **Linealizacion simbolica** — deriva y linealiza en el punto de
   operacion actual (~10s). Muestra Vref0 y que flags de saturacion
   estan activos en el equilibrio.
3. **Reduccion balanceada** — calcula el espectro de valores singulares
   de Hankel, SUGIERE un orden (el mayor salto relativo en el
   espectro), y deja confirmarlo o cambiarlo antes de reducir.
4. **Disenar reguladores (LQI + PI)** — barrido en rejilla sobre el
   modelo YA reducido, para ambos reguladores. Usa `Ts` de los
   parametros.
5. **Bateria de tests** — ejecuta los TRES reguladores (PI, LQI, y un
   rele de banda muerta fijo, no diseniado) sobre: escalon de consigna
   en 4 puntos, falla trifasica de 150ms, y perturbacion de tension de
   red encadenada. Resultados en una ventana con pestanas y graficas
   matplotlib embebidas (Q, Vref, Vt en cada caso).

Cada boton se habilita solo cuando el paso anterior esta completo.
Cambiar los parametros invalida los pasos 2-5 (hay que rehacerlos).

## Sobre el motor de simulacion (`plc_battery.py`)

A diferencia de las simulaciones del paquete original (`04_simulation/`,
donde el regulador se recalcula CONTINUAMENTE dentro de la propia EDO),
aqui el regulador se muestrea y actualiza UNA VEZ POR CICLO Ts, con
Vref mantenido fijo durante todo el ciclo mientras la planta se integra
en continuo -- fiel a como se ejecutaria en un PLC real.

El observador del LQI se actualiza con una discretizacion EXACTA
(exponencial de matriz, metodo de Van Loan), no una aproximacion de
Euler -- necesario porque esta planta tiene polos internos rapidos que
hacen inestable un paso de Euler simple a Ts=100ms.

## Guardar / cargar proyecto

Guarda un unico `.npz` con todos los parametros (JSON embebido) y los
resultados de cada paso completado (matrices de linealizacion,
reduccion, ganancias de los reguladores). La bateria de tests en si NO
se guarda (se recalcula si hace falta verla de nuevo).

## Limitaciones conocidas / no probado a fondo

- El barrido en rejilla del diseno (`workers.py`, `do_design`) usa
  rangos fijos para qy/qi (LQI) y Kp/Ki (PI) -- si el modelo reducido
  cambia mucho (otra maquina, otro AVR), esos rangos podrian no
  contener ninguna combinacion valida. No hay ajuste automatico de la
  rejilla.
- La bateria completa (18 simulaciones: 4 escalones + falla + 4 tramos
  de red, cada uno x3 reguladores) puede tardar varios minutos segun la
  maquina.
- No se ha probado el flujo completo con parametros MUY distintos a los
  valores por defecto (otra maquina real) -- los rangos de los
  spinboxes son generosos pero no exhaustivamente validados.
