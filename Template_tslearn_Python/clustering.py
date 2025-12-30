import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import KShape, TimeSeriesKMeans


def kshape_clustering(X, n_clusters=3):
    model = KShape(n_clusters=n_clusters, random_state=0)
    y_pred = model.fit_predict(X)
    return model, y_pred


def dtw_kmeans_clustering(X, n_clusters=3):
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
    y_pred = model.fit_predict(X)
    return model, y_pred


if __name__ == "__main__":
    X = np.random.rand(100, 50, 1)
    kshape_model, _ = kshape_clustering(X)
    dtw_kmeans_model, _ = dtw_kmeans_clustering(X)

    plt.figure(figsize=(10, 4))
    for centroid in kshape_model.cluster_centers_:
        plt.plot(centroid.ravel())
    plt.title("K-Shape Cluster Centroids")
    plt.show()

    plt.figure(figsize=(10, 4))
    for centroid in dtw_kmeans_model.cluster_centers_:
        plt.plot(centroid.ravel())
    plt.title("DTW K-Means Cluster Centroids")
    plt.show()
