import numpy as np

def gauss_jacobi(*, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int) -> np.array:
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    assert A.shape[0] == A.shape[1], "La matriz A debe ser de tamaño n-by-(n)."

    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=float)
    assert b.shape[0] == A.shape[0], "El vector b debe ser de tamaño n."

    if not isinstance(x0, np.ndarray):
        x0 = np.array(x0, dtype=float)
    assert x0.shape[0] == A.shape[0], "El vector x0 debe ser de tamaño n."

    n = A.shape[0]
    x = x0.copy()
    print(f"i= {0} x: {x.T}")
    for k in range(1, max_iter + 1):
        x_new = np.zeros_like(x0)
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - suma) / A[i, i]
        print(f"i= {k} x: {x_new.T}")
        if np.linalg.norm(x_new - x) < tol:
            print(f"Convergencia alcanzada en {k} iteraciones.")
            return x_new
        x = x_new.copy()
    else:
        print(f"No se alcanzó la convergencia en {max_iter} iteraciones.")
    return x

# Definimos la tolerancia y el número máximo de iteraciones
tol = 1e-2
max_iter = 8

# Matriz
A = np.array([
    [4, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
    [-1, 4, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
    [0, -1, 4, -1, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, -1, 4, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 4, -1, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, -1, 4, -1, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, -1, 4, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 4, -1, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, -1, 4, -1, 0, 0],
    [0, -1, 0, 0, 0, 0, 0, 0, -1, 4, -1, 0],
    [0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 4, -1],
    [0, 0, 0, -1, 0, -1, 0, 0, 0, 0, -1, 4]
], dtype=float)

# Sistema lineal literal b
b = np.array([220, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 220], dtype=float)
x0 = np.zeros(12, dtype=float)

# Resultado
solution = gauss_jacobi(A=A, b=b, x0=x0, tol=tol, max_iter=max_iter)
print("Solución del sistema lineal método de Jacobi:\n", solution)