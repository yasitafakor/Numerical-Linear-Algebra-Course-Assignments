import numpy as np
import sympy as sp
from sympy import init_printing, Eq
s, t = sp.symbols('s t')
init_printing(use_unicode=True)
def r(a, x, B):
    rStar = b - A * X
    r2 = (rStar[0] ** 2) + (rStar[1] ** 2) + (rStar[2] ** 2)
    r2 = sp.simplify(r2)

    for i in range(len(X)):
        fp = sp.diff(r2, X[i])
        X[i] = sp.solveset(Eq(fp, 0), X[i]).args[0]

    return X


print('\nFirst Example:')
A = sp.Matrix([[2, 3], [-1, 2], [1, -4]])
X = sp.Matrix([s, t])
b = sp.Matrix([0, 1, -1])

print(r(A, X, b))

print('\nSecond Example:')
A = sp.Matrix([[1, -1], [2, 3], [1, 1], [1, 2]])
X = sp.Matrix([s, t])
b = sp.Matrix([1, 0, 4, 3])

print(r(A, X, b))

print('\nThird Example:')
s, t, w = sp.symbols('s t w')

A = sp.Matrix([[1, 1, -1], [0, 1, -1], [2, -3, 0], [1, -1, 1]])
X = sp.Matrix([s, t, w])
b = sp.Matrix([1, -1, 1, -1])

print(r(A, X, b))


