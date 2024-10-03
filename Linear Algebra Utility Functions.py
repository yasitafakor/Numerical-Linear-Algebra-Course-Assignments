import numpy as np

def is_symmetric(matrix):
    """Check if the matrix is symmetric."""
    return np.all(matrix == matrix.T)

def calculate_norm(matrix, norm_type='fro'):
    """Calculate various norms of a matrix or vector."""
    if norm_type == '1':
        if matrix.ndim == 1:  # If it's a vector
            return np.sum(np.abs(matrix))
        else:  # If it's a matrix
            return np.sum(np.abs(matrix), axis=0).max()
    elif norm_type == '2':
        return np.sqrt(np.sum(np.square(matrix)))
    elif norm_type == 'fro':
        return np.sqrt(np.sum(np.square(matrix)))
    elif norm_type == 'inf':
        if matrix.ndim == 1:  # If it's a vector
            return np.max(np.abs(matrix))
        else:  # If it's a matrix
            return np.max(np.sum(np.abs(matrix), axis=1))
    else:
        raise ValueError("Invalid norm type specified. Use '1', '2', 'fro', or 'inf'.")


def transpose(matrix):
    """Calculate the transpose of a matrix."""
    rows, cols = matrix.shape
    transposed = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            transposed[j, i] = matrix[i, j]
    return transposed

def eigenvalues(matrix):
    """Find the eigenvalues of a matrix."""
    return np.linalg.eigvals(matrix)

def cholesky_decomposition(matrix):
    """Perform Cholesky decomposition."""
    if not is_symmetric(matrix) or np.any(np.linalg.eigvals(matrix) <= 0):
        raise ValueError("Matrix must be symmetric and positive definite.")

    n = matrix.shape[0]
    L = np.zeros_like(matrix)
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.sqrt(matrix[i, i] - np.sum(L[i, :j] ** 2))
            else:
                L[i, j] = (matrix[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L

def lu_decomposition(matrix):
    """Perform LU decomposition with partial pivoting."""
    n = matrix.shape[0]
    L = np.zeros_like(matrix)
    U = np.zeros_like(matrix)
    
    for i in range(n):
        L[i, i] = 1  
        for j in range(i, n):
            U[i, j] = matrix[i, j] - np.sum(L[i, :i] * U[:i, j])
        
        if U[i, i] == 0:
            raise ValueError("Matrix is singular or ill-conditioned.")
        
        for j in range(i + 1, n):
            L[j, i] = (matrix[j, i] - np.sum(L[j, :i] * U[:i, i])) / U[i, i]
    
    return L, U

def is_idempotent(matrix):
    """Check if the matrix is idempotent."""
    return np.all(matrix @ matrix == matrix)

def is_nilpotent(matrix):
    """Check if the matrix is nilpotent."""
    power = np.eye(matrix.shape[0])
    for _ in range(matrix.shape[0]):
        power = power @ matrix
        if np.all(power == 0):
            return True
    return False

def create_positive_definite_matrix(n):
    """Create a random symmetric positive definite matrix."""
    A = np.random.rand(n, n)
    return np.dot(A, A.T)  # A * A^T is symmetric positive definite
   
# Examples
if __name__ == "__main__":
   
    vector = np.array([1, -2, 3])

    print("1-norm of vector:", calculate_norm(vector, '1'))  
    print("2-norm of vector:", calculate_norm(vector, '2'))  
    print("Frobenius norm of vector:", calculate_norm(vector, 'fro')) 
    print("Infinity norm of vector:", calculate_norm(vector, 'inf'))  

    # Example matrix
    matrix = np.array([[1, -2, 3], [4, -5, 6]])

    print("1-norm of matrix:", calculate_norm(matrix, '1'))  
    print("2-norm of matrix:", calculate_norm(matrix, '2'))  
    print("Frobenius norm of matrix:", calculate_norm(matrix, 'fro')) 
    print("Infinity norm of matrix:", calculate_norm(matrix, 'inf'))  

    matrix_symmetric = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    matrix_non_symmetric = np.array([[1, 2], [3, 4]])
    vector = np.array([1, 2, 3])

    print("Is symmetric:", is_symmetric(matrix_symmetric))  
    print("Is symmetric:", is_symmetric(matrix_non_symmetric))  

    print("Frobenius norm:", calculate_norm(matrix_symmetric, 'fro'))  # Frobenius norm
    print("Infinity norm:", calculate_norm(vector, 'inf'))  # Infinity norm

    print("Transpose of matrix:\n", transpose(matrix_non_symmetric))

    print("Eigenvalues of matrix:\n", eigenvalues(matrix_symmetric))

     # Create a symmetric positive definite matrix
    matrix = create_positive_definite_matrix(3)

    print("Matrix:\n", matrix)

    try:
        L = cholesky_decomposition(matrix)
        print("Cholesky decomposition L:\n", L)
    except ValueError as e:
        print(e)

    non_symmetric_matrix = np.array([[1, 2], [3, 4]])
    print("Testing with non-symmetric matrix:")
    try:
        L = cholesky_decomposition(non_symmetric_matrix)
    except ValueError as e:
        print(e)  

    not_positive_definite_matrix = np.array([[0, 0], [0, 0]])
    print("Testing with not positive definite matrix:")
    try:
        L = cholesky_decomposition(not_positive_definite_matrix)
    except ValueError as e:
        print(e)

    # Example of a valid matrix
    matrix = np.array([[4, 3], [6, 3]])

    try:
        L, U = lu_decomposition(matrix)
        print("L:\n", L)
        print("U:\n", U)
    except ValueError as e:
        print(e)

    # Example of a singular matrix
    singular_matrix = np.array([[1, 2], [2, 4]])

    print("Testing with a singular matrix:")
    try:
        L, U = lu_decomposition(singular_matrix)
        print("L:\n", L)
        print("U:\n", U)
    except ValueError as e:
        print(e)  

    print("Is idempotent:", is_idempotent(np.array([[1, 0], [0, 0]])))  
    print("Is nilpotent:", is_nilpotent(np.array([[0, 0], [0, 0]])))  
