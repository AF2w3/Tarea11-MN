import numpy as np

def gauss_seidel(*, A: np.array, b: np.array, x0: np.array, max_iter: int) -> np.array:

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
    return x

# Definimos el número máximo de iteraciones
max_iter = 3  # Obtención de las dos primeras iteraciones

# Sistema lineal c
A_c = np.array([
    [10, 5, 0, 0],
    [5, 10, -4, 0],
    [-4, 8, 8, -1],
    [0, -3, 5, -11]
], dtype=float)
b_c = np.array([6, 25, -11, -11], dtype=float)
x0_c = np.zeros((4,), dtype=float)

# Resultado del sistema lineal c
print("Sistema c:")
solution_c = gauss_seidel(A=A_c, b=b_c, x0=x0_c, max_iter=max_iter)
print("\nSolución de las 2 primeras iteraciones del sistema lineal c:\n", solution_c)