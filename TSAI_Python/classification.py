import numpy as np
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score


def train_knn_classifier(X, y, neighbors=3, test_size=0.2):
    """
    Train a k-NN classifier with DTW metric for time series classification.

    IMPORTANT: Uses time-based split (not random) to preserve temporal order
    and prevent data leakage in time series data.

    Parameters:
    -----------
    X : array-like
        Time series data (samples, timesteps, features)
    y : array-like
        Labels
    neighbors : int
        Number of neighbors for k-NN
    test_size : float
        Proportion of data to use for testing (default: 0.2)

    Returns:
    --------
    accuracy : float
        Classification accuracy
    """
    # IMPORTANT: For time series, use time-based split, not random split
    # This preserves temporal order and prevents data leakage
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    knn = KNeighborsTimeSeriesClassifier(n_neighbors=neighbors, metric="dtw")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    X = np.random.rand(200, 50, 1)
    y = np.random.randint(0, 2, 200)
    accuracy = train_knn_classifier(X, y)
    print(f"K-NN Classifier Accuracy: {accuracy:.2f}")
