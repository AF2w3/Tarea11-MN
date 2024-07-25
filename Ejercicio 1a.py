import numpy as np

def gauss_jacobi(*, A: np.array, b: np.array, x0: np.array, max_iter: int) -> np.array:
    """Resuelve el sistema de ecuaciones lineales Ax = b mediante el método de Jacobi.

    Parameters
    ----------
    A : np.array
        Matriz de coeficientes del sistema de ecuaciones lineales.
    b : np.array
        Vector de términos independientes del sistema de ecuaciones lineales.
    x0 : np.array
        Vector de aproximación inicial.
    max_iter : int
        Número máximo de iteraciones.

    Returns
    -------
    np.array
        Vector solución del sistema de ecuaciones lineales.
    """
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
        x_new = np.zeros_like(x0)  # prealloc
        for i in range(n):
            suma = sum([A[i, j] * x[j] for j in range(n) if j != i])
            x_new[i] = (b[i] - suma) / A[i, i]

        x = x_new.copy()
        print(f"i= {k} x: {x.T}")

    return x

# Definimos el número máximo de iteraciones
max_iter = 3  # # Obtencion de las dos primeras iteraciones


# Sistema lineal  a
A_a = np.array([[3, -1, 1], [3, 6, 2], [3, 3, 7]], dtype=float)
b_a = np.array([1, 0, 4], dtype=float)
x0_a = np.zeros((3,), dtype=float)

# Resultado del sistema lineal a
print("Sistema a:")
solution_a = gauss_jacobi(A=A_a, b=b_a, x0=x0_a, max_iter=max_iter)
print("\nSolución de las 2 pirmeras iteraciones del sistema lineal es:\n ",solution_a)