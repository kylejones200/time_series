import numpy as np
from tslearn.piecewise import SymbolicAggregateApproximation


def sax_transformation(X, segments=5, alphabet=3):
    sax = SymbolicAggregateApproximation(
        n_segments=segments, alphabet_size_avg=alphabet
    )
    return sax.fit_transform(X)


if __name__ == "__main__":
    X = np.random.rand(100, 50, 1)
    X_sax = sax_transformation(X)
    print("SAX Transformed Data:\n", X_sax)
