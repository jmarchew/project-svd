#Demonstration of my SVD implementation

import numpy as np
from sklearn.decomposition import TruncatedSVD
from svd.mysvd import mysvd


# Three example matrices
examples = {
    "small_square": np.array([[1, 2],
                            [3, 4]], dtype=float),
    "rectangular": np.array([[1, 2, 3],
                            [4, 5, 6]], dtype=float),
    "larger_square": np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]], dtype=float)
}

for name, A in examples.items():
    print(f"\n=== Example: {name} ===")

    # Custom SVD
    sigmas, U_list, V_list = mysvd(A.tolist())

    print("\n--- Custom SVD ---")
    print("Singular values:", sigmas)
    print("V vectors:", V_list)
    print("U vectors:", U_list)

    # scikit-learn SVD
    svd = TruncatedSVD(n_components=min(A.shape))
    svd.fit(A)

    sklearn_sigmas = svd.singular_values_
    sklearn_Vt = svd.components_
    sklearn_V = sklearn_Vt.T  # Convert V^T -> V

    print("\n--- scikit-learn TruncatedSVD ---")
    print("Singular values:", sklearn_sigmas.tolist())
    print("V vectors:", sklearn_V.tolist())
    print("(U is not returned by scikit-learn)")