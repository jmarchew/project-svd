"""
Basic mathematical utilities required for implementing Simple SVD.
Includes vector and matrix operations with input validation and safe checks.

All functions are implemented without using NumPy to illustrate the
mathematical foundations explicitly.
"""

import math

#################################
#########  VALIDATION  ##########
#################################

def ensure_vector(x, name="vector"):
    """
    Validate that input is a numeric vector (list of numbers).
    
    Parameters
    ----------
    x : list
        Input vector.
    name : str
        Display name for error messages.

    Raises
    ------
    TypeError
        If `x` is not a list or contains non-numeric values.
    """
    if not isinstance(x, list):
        raise TypeError(f"{name} must be a list, got {type(x)}")

    for xi in x:
        if not isinstance(xi, (int, float)):
            raise TypeError(
                f"All elements of {name} must be numbers, got element of type {type(xi)}"
            )


def ensure_same_length(a, b):
    """
    Ensure two vectors have equal length.

    Raises
    ------
    ValueError
        If lengths differ.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")


def ensure_matrix(A, name="matrix"):
    """
    Validate that input is a well-formed numeric matrix.

    Parameters
    ----------
    A : list of lists
        Matrix to validate.

    Raises
    ------
    TypeError or ValueError
        For invalid matrix structure or non-numeric values.
    """
    if not isinstance(A, list):
        raise TypeError(f"{name} must be a list (matrix).")

    if len(A) == 0:
        raise ValueError(f"{name} cannot be empty")

    row_length = len(A[0])

    for row in A:
        if not isinstance(row, list):
            raise TypeError(f"Each row of {name} must be a list")
        if len(row) != row_length:
            raise ValueError(f"All rows of {name} must have the same length")
        for value in row:
            if not isinstance(value, (int, float)):
                raise TypeError(f"Matrix elements must be numeric")


#################################
########### VECTORS #############
#################################

def vec_dot(a, b):
    """
    Compute dot product of two vectors.

    Returns
    -------
    float
    """
    ensure_vector(a, "a")
    ensure_vector(b, "b")
    ensure_same_length(a, b)

    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def norm(x):
    """
    Compute L2 norm of a vector.

    Returns
    -------
    float
    """
    ensure_vector(x, "x")
    return math.sqrt(vec_dot(x, x))


def normalize(x):
    """
    Normalize vector to unit length.

    Raises
    ------
    ValueError
        If vector is zero.
    """
    ensure_vector(x, "x")
    n = norm(x)
    if n == 0:
        raise ValueError("Normalization of a zero vector is forbidden!")
    result = []
    for xi in x:
        result.append(xi / n)
    return result


def vector_scalar_mul(a, x):
    """
    Multiply vector by scalar.

    Parameters
    ----------
    a : float
        Scalar value.
    x : list
        Vector.

    Returns
    -------
    list
    """
    if not isinstance(a, (int, float)):
        raise TypeError("Scalar must be numeric")
    ensure_vector(x)

    result = []
    for xi in x:
        result.append(a * xi)
    return result


def vec_add(x, y):
    """
    Add two vectors.

    Returns
    -------
    list
    """
    ensure_vector(x)
    ensure_vector(y)
    ensure_same_length(x, y)
    result = []
    for i in range(len(x)):
        result.append(x[i] + y[i])
    return result


#################################
########### MATRICES ############
#################################

def mat_transpose(A):
    """
    Compute matrix transpose.

    Returns
    -------
    list of lists
    """
    ensure_matrix(A, "A")

    rows = len(A)
    cols = len(A[0])

    T = []
    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(A[i][j])
        T.append(new_row)
    return T


def mat_vector_mul(A, x):
    """
    Multiply matrix by vector.

    Returns
    -------
    list

    Raises
    ------
    ValueError
        If dimensions do not match.
    """
    ensure_matrix(A, "A")
    ensure_vector(x, "x")

    if len(A[0]) != len(x):
        raise ValueError("Matrix column count must match vector length")

    result = []
    for row in A:
        result.append(vec_dot(row, x))
    return result


def matrix_mul(A, B):
    """
    Multiply two matrices (A * B).

    Returns
    -------
    list of lists

    Raises
    ------
    ValueError
        If inner dimensions are incompatible.
    """
    ensure_matrix(A, "A")
    ensure_matrix(B, "B")

    if len(A[0]) != len(B):
        raise ValueError("A columns must equal B rows")

    BT = mat_transpose(B)
    result = []

    for row in A:
        new_row = []
        for col in BT:
            new_row.append(vec_dot(row, col))
        result.append(new_row)

    return result

def mat_sub(A, B):
    """
    Subtract matrices (A - B).

    Returns
    -------
    list of lists

    Raises
    ------
    ValueError
        If matrix dimensions differ.
    """
    ensure_matrix(A, "A")
    ensure_matrix(B, "B")

    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have identical dimensions")

    result = []
    for i in range(len(A)):
        new_row = []
        for j in range(len(A[0])):
            new_row.append(A[i][j] - B[i][j])
        result.append(new_row)
    return result


#################################
########## CONVERGENCE ##########
#################################

def converged(x_old, x_new, eps=1e-9):
    """
    Check whether two vectors are numerically close (change below eps).

    Parameters
    ----------
    x_old, x_new : list
        Vectors to compare.
    eps : float
        Tolerance threshold.

    Returns
    -------
    bool
    """
    ensure_vector(x_old)
    ensure_vector(x_new)
    ensure_same_length(x_old, x_new)

    diff = [x_old[i] - x_new[i] for i in range(len(x_old))]
    return norm(diff) < eps
