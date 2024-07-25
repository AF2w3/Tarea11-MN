import numpy as np

def gauss_seidel(*, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int) -> np.array:

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
    print(f"i= {0} x: {x.T}")
    for k in range(1, max_iter):
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - suma) / A[i, i]
        print(f"i= {k} x: {x.T}")
        if np.linalg.norm(x - x0) < tol:
            break
        x0 = x.copy()
    return x

# Definimos la tolerancia y el número máximo de iteraciones
tol = 1e-3
max_iter = 3  # Un número suficientemente grande para permitir la convergencia

# Sistema lineal a
A_a = np.array([[3, -1, 1], [3, 6, 2], [3, 3, 7]], dtype=float)
b_a = np.array([1, 0, 4], dtype=float)
x0_a = np.zeros((3,), dtype=float)

# Resultado del sistema lineal a
print("Sistema a:")
solution_a = gauss_seidel(A=A_a, b=b_a, x0=x0_a, tol=tol, max_iter=max_iter)
print("\nSolución del sistema lineal a con tolerancia 1e-3:\n", solution_a)