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
import sys
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
