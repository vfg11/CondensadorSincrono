# Publicar en GitHub y compilar el ejecutable de Windows

## 1. Subir el repositorio

No tengo credenciales tuyas ni ningun conector de GitHub disponible en
esta conversacion, asi que estos pasos los haces tu, con tu propia
cuenta:

```bash
cd condenser_new_avr_package
git init
git add .
git commit -m "Estudio de condensador sincrono + GUI PySide6"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
git push -u origin main
```

Si prefieres no usar linea de comandos, puedes crear el repositorio
vacio en github.com y luego arrastrar la carpeta completa en la
interfaz web (funciona para repos pequenos como este).

## 2. Que hace el workflow (`.github/workflows/build-windows.yml`)

Se dispara automaticamente al hacer push a `main` (si tocas
`01_model/`, `02_linearization/`, `05_gui/` o el propio workflow), o
manualmente desde la pestana "Actions" del repositorio en GitHub
("Run workflow").

Pasos que ejecuta, en un runner `windows-latest`:
1. Clona el repositorio
2. Instala Miniconda + crea el entorno de `environment.yml` (usa
   conda-forge especificamente porque `slycot` no siempre tiene wheel
   precompilada para Windows via pip puro -- conda-forge si la tiene
   de forma fiable)
3. Verifica que todas las dependencias importan correctamente
4. Compila con Nuitka en modo `--onefile` (un unico .exe autocontenido)
5. Sube el .exe como artefacto descargable

## 3. Donde encontrar el ejecutable

Una vez el workflow termine (pestana "Actions" del repo, entra en la
ejecucion mas reciente): al final de la pagina, en "Artifacts",
aparece `CondensadorSincronoGUI-windows` -- descargalo (es un .zip que
contiene el .exe).

## 4. Primera ejecucion: cosas a verificar

Esto no lo he podido probar en un Windows real (solo he verificado
partes en Linux, que no es la plataforma final):
- El primer arranque del .exe en modo `--onefile` puede tardar unos
  segundos mas de lo normal (se autoextrae a una carpeta temporal cada
  vez que se ejecuta) -- no es que se haya colgado.
- Si Windows Defender/SmartScreen avisa de "editor desconocido" (normal
  para un .exe sin firmar), hay que darle a "mas informacion" ->
  "ejecutar de todas formas".
- Si el workflow falla en el paso de Nuitka, el log completo queda en
  la propia ejecucion de Actions -- lo mas probable, si falla algo, es
  en la deteccion de plugins de PySide6/matplotlib, ajustable en
  `.github/workflows/build-windows.yml` (banderas `--enable-plugin`,
  `--include-package-data`).

## 5. Compilar localmente en Windows (opcional, para probar antes de CI)

Si tienes acceso a una maquina Windows y quieres probar sin pasar por
GitHub Actions:

```powershell
# Con Miniconda/Anaconda ya instalado:
conda env create -f environment.yml
conda activate condenser-gui

python -m nuitka `
  --mode=onefile `
  --enable-plugin=pyside6 `
  --windows-console-mode=disable `
  --mingw64 `
  --assume-yes-for-downloads `
  --output-dir=dist `
  --output-filename=CondensadorSincronoGUI.exe `
  --include-package-data=matplotlib `
  05_gui/main_app.py
```

## 6. Verificado / no verificado (para que sepas donde esta el riesgo real)

**Verificado en este entorno (Linux, no es la plataforma final)**:
- Los 4 archivos con manipulacion de `sys.path` (`main_app.py`,
  `workers.py`, `params_registry.py`, `linearize_condenser.py`) se
  revisaron para que la resolucion de rutas no dependa de trucos
  fragiles al compilar -- protegida con try/except, nunca puede hacer
  que la app falle en el arranque.
- La app sigue funcionando identica en ejecucion normal
  (`python3 main_app.py`) tras todos estos cambios.
- Nuitka, con el PYTHONPATH equivalente al que usara el workflow,
  encuentra y procesa correctamente `genqec_model`, `controls`,
  `linearize_condenser` y todas sus dependencias (incluyendo sympy) sin
  ningun error de modulo no encontrado -- verificado dejandolo avanzar
  varios cientos de modulos en la fase de optimizacion.

**NO verificado (solo puede confirmarse en el runner Windows real)**:
- Que la compilacion complete de principio a fin (no pude terminar una
  compilacion completa en este sandbox Linux -- sympy por si solo tiene
  cientos de submodulos, y ademas necesito el resultado en Windows, no
  Linux).
- Que el .exe resultante arranque y funcione correctamente en un
  Windows real.
- Que `slycot` se instale sin problemas via conda-forge en el runner
  `windows-latest` concreto (la documentacion y la comunidad lo senalan
  como el camino fiable, pero no es lo mismo que haberlo visto pasar).

Te recomendaria, en la primera ejecucion del workflow, revisar el log
completo aunque termine en verde, para confirmar que no hay avisos
raros de plugins o DLLs faltantes.
