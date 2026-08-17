# -*- mode: python ; coding: utf-8 -*-
#
# main_app.spec
# ===============
# Empaqueta la GUI (05_gui/main_app.py) con PyInstaller. Retomado tras
# varias dificultades especificas con Nuitka y el bundling de las DLLs
# de MKL/OpenMP bajo conda-forge en Windows (ver chat para el
# diagnostico completo) -- PyInstaller tiene hooks dedicados y maduros
# para numpy/scipy que deberian cubrir esto de fabrica.
#
# pathex apunta a las 3 carpetas necesarias (01_model, 02_linearization,
# 05_gui) para que las importaciones "planas" (import genqec_model,
# etc, sin prefijo de paquete) se resuelvan en el analisis estatico --
# el codigo fuente ya tiene ademas su propio try/except de sys.path
# para la ejecucion normal, asi que esto es un refuerzo, no la unica
# via.
#
# Ejecutar con: pyinstaller main_app.spec

import os

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(SPEC)))

block_cipher = None

a = Analysis(
    [os.path.join(REPO_ROOT, '05_gui', 'main_app.py')],
    pathex=[
        os.path.join(REPO_ROOT, '01_model'),
        os.path.join(REPO_ROOT, '02_linearization'),
        os.path.join(REPO_ROOT, '05_gui'),
    ],
    binaries=[],
    datas=[],
    hiddenimports=[
        'slycot',
        'slycot._wrapper',
        'control',
        'scipy.linalg',
        'scipy.integrate',
        'scipy.special',
        'matplotlib.backends.backend_qtagg',
        'PySide6.QtSvg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pandas', 'openpyxl', 'gi', 'cffi', 'pytest', 'IPython', 'tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CondensadorSincronoGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
