import numpy as np

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

# Verificamos si la matriz es diagonalmente dominante
def es_diagonalmente_dominante(A):
    n = A.shape[0]
    for i in range(n):
        suma = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= suma:
            return False
    return True

print("La matriz es diagonalmente dominante:", es_diagonalmente_dominante(A))