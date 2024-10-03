import math

import numpy as np
import time

import numpy.linalg

start = time.time()
mhk = 5 * (10 ** (-14))


def Thomas(a, b, c, d):
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


def SOR(d, l, u, w, b, x):

    result = x
    Msor = (-1)*(np.matmul(numpy.linalg.inv(d + w * l), (w - 1) * d + w * u))
    Csor = np.matmul(w * (np.linalg.inv(d + w * l)), b)

   # while ((np.linalg.norm(b - np.matmul((l + d + u), result), np.inf)) / np.linalg.norm(b, np.inf)) > mhk:
    result = np.matmul(Msor,result) + Csor

    return result


n = 2000
A = np.zeros((n, n))
L = np.zeros((n, n))
U = np.zeros((n, n))
D = np.zeros((n, n))
d = np.zeros(n)

for i in range(0, n):

    d[i] = i + 1
    A[i, i] = 2
    D[i, i] = 2
    if i != n - 1:
        A[i + 1, i] = 1
        L[i + 1, i] = 1
    if i != 0:
        A[i - 1, i] = 1
        U[i - 1, i] = 1

a = np.zeros(n)
b = np.zeros(n)
c = np.zeros(n)

for i in range(0, n):
    b[i] = 2

for i in range(0, n - 1):
    a[i] = 1

for i in range(0, n - 1):
    c[i] = 1

# for i in Thomas(a, b, c, d):
# print(i)
end = time.time()
print(Thomas(a, b, c, d))
print("Thomas Time:")
print(end - start)

x = np.zeros((n, 1))

Mj = np.zeros((n, n))
Mj = (-1) * (np.matmul(np.linalg.inv(D), L + U))
w, v = np.linalg.eig(Mj)
maximum = w.max()
n = 2 / (1 + math.sqrt(1 - maximum ** 2))
print(n)

s = SOR(D, L, U, n, d, x)
print(s)
