# Linear Algebra and Numerical Methods Projects

This repository contains a collection of projects focused on linear algebra and numerical methods, implemented in Python. The projects cover a wide range of techniques, from eigenvalue computation to image compression, showcasing the power of linear algebra in various applications.

## Projects Overview

### 1. Power Method for Finding Maximum and Minimum Eigenvalue

**Overview:**
This project implements algorithms for finding the maximum and minimum eigenvalues and corresponding eigenvectors of a symmetric tridiagonal matrix using the Power Method and the Inverse Power Method.

**Description:**
- **Power Method:** Iteratively approximates the maximum eigenvector and eigenvalue of a given matrix \( A \).
- **Inverse Power Method:** Finds the minimum eigenvector and eigenvalue of \( A \) by applying the inverse of \( A \) to an initial guess.
- The matrix \( A \) is a symmetric tridiagonal matrix with diagonal elements of 4 and off-diagonal elements of -1.

---

### 2. Partial

**Overview:**
This project demonstrates the use of symbolic computation to solve systems of equations using the SymPy library, focusing on minimizing the residuals of the equations.

**Description:**
- **Residual Calculation:** Computes the residual vector \( r^* = b - Ax \).
- **Differentiation:** Differentiates the sum of squares of residuals with respect to each variable.
- **Finding Solutions:** Sets the derivative to zero and solves for each variable to minimize the residuals.

---

### 3. Comparing Thomas and SOR

**Overview:**
This project implements numerical methods for solving systems of linear equations using the Thomas algorithm for tridiagonal matrices and the Successive Over-Relaxation (SOR) method.

**Description:**
- **Thomas Algorithm:** Efficiently solves tridiagonal systems in \( O(n) \) time complexity.
- **SOR Method:** An iterative method that improves convergence using an optimal relaxation factor.
- Measures performance and execution time for large systems.

---

### 4. Image Compression Using Singular Value Decomposition (SVD)

**Overview:**
This project demonstrates image compression using Singular Value Decomposition (SVD), enabling efficient storage and transmission of images.

**Description:**
- **Image Loading:** Loads the image using the Pillow library and converts it into a NumPy array.
- **Normalization:** Normalizes pixel values to the range [0, 1].
- **SVD Compression:** Decomposes the image into three matrices and reconstructs it using a specified number of singular values.
- **Saving and Displaying:** Compressed images are saved and displayed for visual inspection.

---

### 5. Linear Algebra Utility Functions

**Overview:**
This project contains a collection of utility functions for various linear algebra operations implemented in Python.

**Functions:**
1. `is_symmetric(matrix)`: Checks if a matrix is symmetric.
2. `calculate_norm(matrix, norm_type='fro')`: Calculates various norms (1-norm, 2-norm, Frobenius norm, and infinity norm).
3. `transpose(matrix)`: Calculates the transpose of a matrix.
4. `eigenvalues(matrix)`: Finds the eigenvalues of a matrix.
5. `cholesky_decomposition(matrix)`: Performs Cholesky decomposition on a symmetric positive definite matrix.
6. `lu_decomposition(matrix)`: Performs LU decomposition with partial pivoting.
7. `is_idempotent(matrix)`: Checks if a matrix is idempotent.
8. `is_nilpotent(matrix)`: Checks if a matrix is nilpotent.
9. `create_positive_definite_matrix(n)`: Creates a random symmetric positive definite matrix.
