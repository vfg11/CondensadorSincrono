"""
diag_numpy_scipy.py
======================
Diagnostico minimo: prueba las mismas operaciones de algebra lineal
(autovalores, ecuacion de Lyapunov, exponencial de matriz) que usa la
app real, pero SIN sympy/PySide6/matplotlib -- para ver si el problema
es generico de numpy/scipy bajo esta combinacion conda+Nuitka, o algo
mas especifico del resto de la app. Compila mucho mas rapido (sin
sympy ni PySide6 de por medio).
"""
import os
import sys
# DEBE ir antes de importar numpy/scipy. Fix documentado para el
# conflicto "OMP: Error #15: Initializing libiomp5md.dll, but found
# libiomp5md.dll already initialized" -- ocurre cuando mas de un
# runtime OpenMP se carga en el mismo proceso (aqui: el de Intel/MKL
# que trae numpy/scipy de conda, y posiblemente el de MinGW64/GCC que
# usa Nuitka para compilar). Ver chat para el diagnostico completo.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Registrar explicitamente la carpeta con las DLLs de MKL/OpenMP que el
# workflow copia a mano junto al ejecutable (ver
# .github/workflows/diag-numpy-scipy.yml, --include-data-files) --
# necesario porque ni --include-data-dir ni los plugins dll-files/
# data-files de Nuitka reconocieron correctamente libiomp5md.dll para
# esta combinacion concreta conda-forge+MKL (ver chat, verificado con
# Process Monitor). os.add_dll_directory es la API oficial de Windows
# para esto (Python 3.8+).
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    _exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    _extra_dll_dir = os.path.join(_exe_dir, 'extra_dlls')
    if os.path.isdir(_extra_dll_dir):
        os.add_dll_directory(_extra_dll_dir)
import numpy as np
import scipy
import scipy.linalg as sla

print(f"Python: {sys.version}")
print(f"numpy: {np.__version__}")
print(f"scipy: {scipy.__version__}")
print()

np.random.seed(0)
A = np.random.randn(5, 5)
B = A @ A.T + np.eye(5)

print("Probando numpy.linalg.eig ...")
w, v = np.linalg.eig(A)
print("  OK, primeros autovalores:", w[:2])

print("Probando scipy.linalg.solve (sistema lineal) ...")
x = sla.solve(B, np.ones(5))
print("  OK, x[:2]=", x[:2])

print("Probando scipy.linalg.solve_lyapunov ...")
X = sla.solve_lyapunov(A - 5 * np.eye(5), -B)
print("  OK, shape:", X.shape)

print("Probando scipy.linalg.expm ...")
E = sla.expm(A * 0.1)
print("  OK, shape:", E.shape)

print("Probando numpy.linalg.svd ...")
U, S, Vt = np.linalg.svd(A)
print("  OK, valores singulares:", S.round(4))

print()
print("TODAS LAS PRUEBAS PASARON SIN FALLO.")
input("Pulsa Enter para salir...")
