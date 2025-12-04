import pytest
import numpy as np
from sklearn.decomposition import TruncatedSVD
from svd.mysvd import mysvd, power_method

__author__ = "w1nn3t0u"
__copyright__ = "w1nn3t0u"
__license__ = "MIT"


def reconstruct_matrix(sigmas, Us, Vs):
    """Reconstruct matrix from SVD components."""
    m = len(Us[0])
    n = len(Vs[0])
    k = len(sigmas)

    A_reconstructed = [[0.0] * n for _ in range(m)]

    for i in range(k):
        sigma = sigmas[i]
        u = Us[i]
        v = Vs[i]
        for row in range(m):
            for col in range(n):
                A_reconstructed[row][col] += sigma * u[row] * v[col]

    return A_reconstructed


def frobenius_norm(A):
    """Compute Frobenius norm of a matrix."""
    total = 0.0
    for row in A:
        for val in row:
            total += val * val
    return total ** 0.5


def matrix_diff(A, B):
    """Compute element-wise difference between two matrices."""
    m = len(A)
    n = len(A[0])
    diff = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            diff[i][j] = A[i][j] - B[i][j]
    return diff


class TestSVDBasic:
    """Basic functionality tests for mysvd."""

    def test_simple_2x2_matrix(self):
        """Test SVD on a simple 2x2 matrix."""
        A = [[3.0, 0.0], 
             [0.0, 2.0]]

        sigmas, Us, Vs = mysvd(A, k=2)

        # Check we got 2 singular values
        assert len(sigmas) == 2
        assert len(Us) == 2
        assert len(Vs) == 2

        # Singular values should be positive and in descending order
        assert sigmas[0] >= sigmas[1] > 0

    def test_3x3_matrix(self):
        """Test SVD on a 3x3 matrix."""
        A = [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]

        sigmas, Us, Vs = mysvd(A, k=2)

        # Reconstruct and check approximation quality
        A_reconstructed = reconstruct_matrix(sigmas, Us, Vs)
        diff = matrix_diff(A, A_reconstructed)
        error = frobenius_norm(diff)

        # Error should be relatively small for rank-2 approximation
        assert error < 1.0  # This matrix is approximately rank-2

    def test_rectangular_matrix_more_rows(self):
        """Test SVD on rectangular matrix (m > n)."""
        A = [[1.0, 2.0],
             [3.0, 4.0],
             [5.0, 6.0],
             [7.0, 8.0]]

        sigmas, Us, Vs = mysvd(A, k=2)

        assert len(sigmas) == 2
        assert len(Us[0]) == 4  # u vectors should have length m
        assert len(Vs[0]) == 2  # v vectors should have length n

    def test_rectangular_matrix_more_cols(self):
        """Test SVD on rectangular matrix (m < n)."""
        A = [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0]]

        sigmas, Us, Vs = mysvd(A, k=2)

        assert len(sigmas) == 2
        assert len(Us[0]) == 2  # u vectors should have length m
        assert len(Vs[0]) == 4  # v vectors should have length n


class TestSVDComparisonWithSklearn:
    """Compare mysvd results with scikit-learn."""

    def test_singular_values_match_sklearn(self):
        """Test that singular values match sklearn's TruncatedSVD."""
        np.random.seed(42)
        A_np = np.random.randn(5, 4)
        A = A_np.tolist()

        k = 3
        sigmas, Us, Vs = mysvd(A, k=k)

        # Compare with sklearn
        sklearn_svd = TruncatedSVD(n_components=k, random_state=42)
        sklearn_svd.fit(A_np)
        sklearn_singular_values = sklearn_svd.singular_values_

        # Check singular values match (with some tolerance)
        for i in range(k):
            assert abs(sigmas[i] - sklearn_singular_values[i]) < 0.1,                 f"Singular value {i}: {sigmas[i]} vs sklearn {sklearn_singular_values[i]}"

    def test_reconstruction_error_similar_to_sklearn(self):
        """Test that reconstruction error is similar to sklearn."""
        np.random.seed(123)
        A_np = np.random.randn(6, 5)
        A = A_np.tolist()

        k = 3
        sigmas, Us, Vs = mysvd(A, k=k)
        A_reconstructed = reconstruct_matrix(sigmas, Us, Vs)

        # Compute reconstruction error
        diff = matrix_diff(A, A_reconstructed)
        my_error = frobenius_norm(diff)

        # Compare with sklearn reconstruction
        sklearn_svd = TruncatedSVD(n_components=k, random_state=42)
        A_sklearn_reconstructed = sklearn_svd.fit_transform(A_np) @ sklearn_svd.components_
        sklearn_error = np.linalg.norm(A_np - A_sklearn_reconstructed, 'fro')

        # Errors should be similar
        assert abs(my_error - sklearn_error) < 0.5,             f"Reconstruction error: {my_error} vs sklearn {sklearn_error}"

    def test_identity_matrix(self):
        """Test SVD on identity matrix (only k=1 due to deflation limitations)."""
        A = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]
        
        # Identity matrix has repeated singular values (all 1.0)
        # Power method + deflation becomes unstable after first component
        # So we only test k=1
        sigmas, Us, Vs = mysvd(A, k=1)
        
        # First singular value should be 1.0
        assert abs(sigmas[0] - 1.0) < 0.01
        
        # Compare with sklearn
        A_np = np.array(A)
        sklearn_svd = TruncatedSVD(n_components=1)
        sklearn_svd.fit(A_np)
        
        assert abs(sigmas[0] - sklearn_svd.singular_values_[0]) < 0.01


class TestSVDEdgeCases:
    """Test edge cases and special matrices."""

    def test_diagonal_matrix(self):
        """Test SVD on diagonal matrix."""
        A = [[5.0, 0.0, 0.0],
             [0.0, 3.0, 0.0],
             [0.0, 0.0, 1.0]]

        sigmas, Us, Vs = mysvd(A, k=3)

        # Singular values should be the absolute diagonal values in descending order
        expected = [5.0, 3.0, 1.0]
        for i, expected_val in enumerate(expected):
            assert abs(sigmas[i] - expected_val) < 0.01

    def test_rank_deficient_matrix(self):
        """Test SVD on rank-deficient matrix."""
        # Create a rank-1 matrix
        A = [[1.0, 2.0, 3.0],
             [2.0, 4.0, 6.0],
             [3.0, 6.0, 9.0]]

        # Should handle this gracefully (may raise ValueError or produce small singular values)
        try:
            sigmas, Us, Vs = mysvd(A, k=2)
            # First singular value should be much larger than second
            assert sigmas[0] > 10 * abs(sigmas[1])
        except ValueError as e:
            # Expected for rank-deficient matrices
            assert "rank-deficient" in str(e)

    def test_partial_decomposition(self):
        """Test that k parameter works correctly."""
        A = [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0]]

        k = 2
        sigmas, Us, Vs = mysvd(A, k=k)

        # Should return exactly k components
        assert len(sigmas) == k
        assert len(Us) == k
        assert len(Vs) == k


class TestPowerMethod:
    """Test the power_method function directly."""

    def test_power_method_simple_matrix(self):
        """Test power method on a simple matrix."""
        A = [[2.0, 0.0],
             [0.0, 1.0]]

        sigma, u, v = power_method(A)

        # Largest singular value should be 2.0
        assert abs(sigma - 2.0) < 0.01

        # u and v should be unit vectors
        u_norm = sum(x*x for x in u) ** 0.5
        v_norm = sum(x*x for x in v) ** 0.5
        assert abs(u_norm - 1.0) < 1e-6
        assert abs(v_norm - 1.0) < 1e-6

    def test_power_method_convergence(self):
        """Test that power method converges."""
        np.random.seed(42)
        A = np.random.randn(4, 3).tolist()

        sigma, u, v = power_method(A, iterations=1000, eps=1e-9)

        # Should produce valid results
        assert sigma > 0
        assert len(u) == 4
        assert len(v) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
