"""
Minimal SVD (power method + deflation).

"""

from svd.mymath import (
    normalize,
    norm,
    mat_sub,
    mat_transpose,
    mat_vector_mul,
    matrix_mul,
    converged,
    ensure_matrix,
    ensure_vector,
)
from sklearn.decomposition import TruncatedSVD
import numpy as np


def power_method(A, iterations=1000, eps=1e-9):
    """
    Compute dominant singular value and corresponding singular vectors
    using the power method on A^T A.
    """
    # validate matrix
    ensure_matrix(A)

    n = len(A[0])

    # starting vector (non-zero)
    x = [1.0] * n
    ensure_vector(x)
    x = normalize(x)

    for _ in range(iterations):
        # x_new = (A^T A) x
        ATA_x = mat_vector_mul(matrix_mul(mat_transpose(A), A), x)
        ensure_vector(ATA_x)
        x_new = normalize(ATA_x)

        if converged(x, x_new, eps):
            break
        x = x_new

    Ax = mat_vector_mul(A, x)
    sigma = norm(Ax)

    if sigma == 0:
        # rank-deficient or zero matrix
        raise ValueError("Computed singular value is zero (matrix may be rank-deficient).")

    u = normalize(Ax)
    v = x_new
    return sigma, u, v


def mysvd(A, k=None):
    """
    Simple SVD by iteratively finding dominant singular triplets and deflating.
    Returns lists: (sigmas, Us, Vs)
    """
    ensure_matrix(A)
    m = len(A)
    n = len(A[0])

    # working copy
    A_copy = []
    for row in A:
        new_row = []
        for val in row:
            new_row.append(val)
        A_copy.append(new_row)

    if k is None:
        k = min(m, n)

    sigmas = []
    Us = []
    Vs = []

    for _ in range(k):
        sigma, u, v = power_method(A_copy)

        sigmas.append(sigma)
        Us.append(u)
        Vs.append(v)

        # deflation: A' = A - sigma * (u v^T)
        rank1 = []
        for i in range(m):
            new_row = []
            for j in range(n):
                new_row.append(sigma * u[i] * v[j])
            rank1.append(new_row)

        A_copy = mat_sub(A_copy, rank1)

    return sigmas, Us, Vs
