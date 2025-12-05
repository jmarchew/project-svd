import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD

from svd.mysvd import mysvd


def demo_simple_matrix():
    print("\n=== DEMO: matrix 3×3 ===")

    A = np.array([
        [3, 1, 1],
        [-1, 3, 1],
        [1, 1, 3]
    ], dtype=float)

    print("Macierz A:")
    print(A)

    S, U, V = mysvd(A.tolist())

    print("\nU (left singular vectors):")
    print(np.array(U))

    print("\nSingular values S:")
    print(S)

    print("\nV (right singular vectors):")
    print(np.array(V))

    # Reconstruction U * S * V
    A_reconstructed = np.array(U) @ np.diag(S) @ np.array(V)
    print("\nA reconstructed:")
    print(A_reconstructed)


def demo_iris_dataset():
    print("\n=== DEMO: SVD on IRIS dataset ===")

    data = load_iris()
    X = data.data  

    print("Original datam matrix X shape:", X.shape)

    
    X_centered = X - X.mean(axis=0)

    
    S, U, V = mysvd(X_centered.tolist())

    print("\nTop 4 singular values:")
    print(S)

    S = np.array(S)  # conversion

    explained_variance = (S**2) / np.sum(S**2)
    print("\nExplained variance (%):")
    print(explained_variance * 100)

    #casting on first 2 components
    X_projected = X_centered @ np.array(V).T[:, :2]
    
    print("\n=== Comparison with sklearn TruncatedSVD ===")

    # sklearn SVD (only V and singular values)
    sklearn_svd = TruncatedSVD(n_components=4)
    sklearn_svd.fit(X_centered)

    print("\nSingular values (sklearn):")
    print(sklearn_svd.singular_values_)

    # comparison of singular values
    print("\nSingular values comparison (mysvd vs sklearn):")
    for a, b in zip(S, sklearn_svd.singular_values_):
        print(f"mysvd: {a:.6f}   sklearn: {b:.6f}   difference: {abs(a-b):.6f}")

    # sklearn components_  is V in mysvd
    V_sklearn = sklearn_svd.components_

    print("\nV matrice comparison (2 columns):")
    print("V (mysvd):")
    print(np.array(V)[:2])
    print("\nV (sklearn):")
    print(V_sklearn[:2])

    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=data.target)
    plt.title("Iris projected")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


if __name__ == "__main__":
    demo_simple_matrix()
    demo_iris_dataset()
