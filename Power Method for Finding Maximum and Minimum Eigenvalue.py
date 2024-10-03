import math

import numpy as np
import time

import numpy.linalg

n = 3
A = np.zeros((n, n), dtype=np.float64)

for i in range(0, n):

    A[i, i] = 4
    if i != 0:
        A[i - 1, i] = -1
    if i != n - 1:
        A[i + 1, i] = -1


V = np.zeros((n, 1))
V[1] = 1

for i in range(0, 5):
    V = np.matmul(A, V)
    maxEigenvalue = V
    V = V/np.linalg.norm(V, np.inf)


print('Maximum eigenvector:')
print(V)
print('\n')
print('Maximum eigenvalue:')
print(np.linalg.norm(maxEigenvalue, np.inf))
print('\n')


V2 = np.zeros((n, 1))
V2[1] = 1

A2 = np.linalg.inv(A)

for i in range(0, 5):
    V2 = np.matmul(A2, V2)
    minEigenvalue = V2
    V2 = V2/np.linalg.norm(V2, np.inf)


print('Minimum eigenvector:')
print(V2)
print('\n')
print('Minimum eigenvalue:')
print(1/np.linalg.norm(minEigenvalue, np.inf))