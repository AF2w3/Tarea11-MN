import numpy as np

def gauss_jacobi(*, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int) -> np.array:

    # --- Validación de los argumentos de la función ---
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    assert A.shape[0] == A.shape[1], "La matriz A debe ser de tamaño n-by-(n)."

    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=float)
    assert b.shape[0] == A.shape[0], "El vector b debe ser de tamaño n."

    if not isinstance(x0, np.ndarray):
        x0 = np.array(x0, dtype=float)
    assert x0.shape[0] == A.shape[0], "El vector x0 debe ser de tamaño n."

    # --- Algoritmo ---
    n = A.shape[0]
    x = x0.copy()
    for k in range(1, max_iter + 1):
        x_new = np.zeros_like(x0)  # prealloc
        for i in range(n):
            suma = sum([A[i, j] * x[j] for j in range(n) if j != i])
            x_new[i] = (b[i] - suma) / A[i, i]

        if np.linalg.norm(x_new - x) < tol:
            return x_new

        x = x_new.copy()
    return x

# Definimos la tolerancia y el número máximo de iteraciones
tol = 1e-3
max_iter = 25  # Demostrar que falla después de 25 iteraciones

# Sistema lineal literal a
A = np.array([[2, -1, 1], [1, 2, 2], [-1, -1, 2]], dtype=float)
b = np.array([-1, 4, -5], dtype=float)
x0 = np.zeros((3,), dtype=float)

# Resultado del sistema lineal
solution = gauss_jacobi(A=A, b=b, x0=x0, tol=tol, max_iter=max_iter)
print("Solución después de 25 iteraciones utilizando el método de Jacobi:\n", solution)