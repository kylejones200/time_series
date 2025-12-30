# Time Series Similarity and Clustering

## Chapter Overview

Traditional machine learning often assumes that data points are independent, but time series data has a temporal structure that requires specialized approaches. This chapter explores methods for measuring similarity between time series, clustering time series data, and classifying time series using distance-based methods.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand Dynamic Time Warping (DTW) and its applications
- Measure similarity between time series using DTW distance
- Cluster time series using K-Shape and DTW-based K-Means
- Classify time series using k-Nearest Neighbors with DTW
- Apply time series preprocessing techniques (resampling, normalization)
- Use Symbolic Aggregate Approximation (SAX) for feature extraction

### Why Time Series Similarity Matters

Time series similarity is fundamental to many applications:

- **Anomaly Detection**: Find time series that are dissimilar to normal patterns
- **Clustering**: Group similar time series together
- **Classification**: Classify time series based on similarity to labeled examples
- **Pattern Recognition**: Identify recurring patterns in historical data
- **Recommendation Systems**: Find similar time series for recommendations

---

## 19.1 Introduction to Time Series Distance Metrics

Traditional distance metrics like Euclidean distance assume that time series are aligned—that is, the value at time `t` in one series corresponds to the value at time `t` in another series. However, time series often have:

- **Different speeds**: One pattern might occur faster than another
- **Phase shifts**: Similar patterns might be offset in time
- **Different lengths**: Time series might have different durations

**Dynamic Time Warping (DTW)** addresses these challenges by finding the optimal alignment between two time series, allowing for non-linear warping of the time axis.

---

## 19.2 Dynamic Time Warping (DTW)

DTW is a technique for measuring similarity between two time series that may vary in speed or timing. It finds the optimal alignment between sequences by "warping" the time axis.

### Understanding DTW

DTW works by:

1. **Finding Optimal Alignment**: It finds the best way to align two time series
2. **Allowing Non-linear Warping**: Unlike Euclidean distance, DTW can match points that are not at the same time index
3. **Computing Distance**: The distance is the sum of distances between aligned points

### Basic DTW Example

```python
from tslearn.metrics import dtw
import numpy as np

# Two synthetic time series
ts1 = np.array([1, 2, 3, 4, 5])
ts2 = np.array([2, 3, 4, 5, 6])

# Compute DTW distance
distance = dtw(ts1, ts2)
print(f"DTW Distance: {distance:.2f}")
```

**Key Properties of DTW:**

- **Symmetric**: `dtw(A, B) = dtw(B, A)`
- **Non-negative**: Always returns a non-negative value
- **Not a true metric**: Doesn't always satisfy the triangle inequality
- **Computational Cost**: O(n×m) where n and m are series lengths

### When to Use DTW

- **Variable Speed**: When patterns occur at different speeds
- **Phase Shifts**: When similar patterns are offset in time
- **Different Lengths**: When time series have different durations
- **Shape Similarity**: When you care about shape rather than exact timing

### Limitations of DTW

- **Computational Cost**: Can be slow for long time series
- **Sensitivity to Noise**: Can be affected by outliers
- **Not Interpretable**: The warping path is not always intuitive

---

## 19.3 Time Series Preprocessing

Before applying similarity measures or clustering, we often need to preprocess time series data.

### Resampling Time Series

Time series may have different lengths or sampling rates. Resampling adjusts them to a consistent length:

```python
import numpy as np
from tslearn.preprocessing import TimeSeriesResampler

# Create a synthetic univariate time series dataset
X = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7]
])

print(X.shape)  # (3 samples, 5 timesteps)

# Resample to 10 timesteps
resampler = TimeSeriesResampler(sz=10)
X_resampled = resampler.fit_transform(X)

print(X_resampled.shape)  # (3 samples, 10 timesteps)
```

**Use Cases for Resampling:**

- **Normalization**: Make all series the same length
- **Downsampling**: Reduce computational cost
- **Upsampling**: Increase resolution for analysis

### Normalizing Time Series

Normalization standardizes time series to have zero mean and unit variance:

```python
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

scaler = TimeSeriesScalerMeanVariance()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

**Why Normalize?**

- **Scale Independence**: Makes distance measures independent of scale
- **Comparability**: Allows comparison of series with different magnitudes
- **Clustering**: Essential for clustering algorithms that are sensitive to scale

---

## 19.4 Time Series Clustering

Clustering groups similar time series together without labeled data. This is useful for:

- **Pattern Discovery**: Finding common patterns in historical data
- **Segmentation**: Dividing time series into groups
- **Anomaly Detection**: Identifying series that don't fit any cluster

### K-Shape Clustering

K-Shape is a clustering algorithm specifically designed for time series. It clusters based on **shape similarity** rather than point-wise similarity.

```python
from tslearn.clustering import KShape
import matplotlib.pyplot as plt
import numpy as np

# Synthetic time series dataset
X = np.random.rand(100, 50, 1)  # 100 samples, 50 timesteps

# Apply K-Shape clustering
kshape = KShape(n_clusters=3, random_state=0)
y_pred = kshape.fit_predict(X)

# Plot cluster centroids
for centroid in kshape.cluster_centers_:
    plt.plot(centroid.ravel())

plt.title("K-Shape Cluster Centroids")
plt.show()
```

**Key Features of K-Shape:**

- **Shape-Based**: Focuses on shape similarity, not exact values
- **Scale Invariant**: Works well with normalized data
- **Interpretable**: Centroids represent typical patterns in each cluster

**When to Use K-Shape:**

- **Shape Patterns**: When you care about shape, not magnitude
- **Normalized Data**: When time series are normalized
- **Interpretability**: When you need interpretable cluster centroids

### DTW-Based K-Means Clustering

DTW K-Means uses Dynamic Time Warping as the distance metric in K-Means clustering:

```python
from tslearn.clustering import TimeSeriesKMeans

# Apply DTW-based K-Means
dtw_kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0)
y_pred = dtw_kmeans.fit_predict(X)

# Plot cluster centroids
for centroid in dtw_kmeans.cluster_centers_:
    plt.plot(centroid.ravel())

plt.title("DTW K-Means Cluster Centroids")
plt.show()
```

**DTW K-Means vs. K-Shape:**

- **DTW K-Means**: Uses DTW distance, good for variable-speed patterns
- **K-Shape**: Uses shape-based distance, good for normalized shape patterns
- **Computational Cost**: DTW K-Means is typically slower

**Choosing Between Methods:**

- Use **K-Shape** when series are normalized and you care about shape
- Use **DTW K-Means** when series have different speeds or phase shifts

---

## 19.5 Time Series Classification

Time series classification assigns labels to time series based on their characteristics. k-Nearest Neighbors (kNN) with DTW is a simple but effective approach.

### k-Nearest Neighbors with DTW

```python
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Create synthetic dataset
X = np.random.rand(200, 50, 1)  # 200 samples, 50 timesteps
y = np.random.randint(0, 2, 200)  # Binary labels

# Split into training and testing sets
# IMPORTANT: For time series, use time-based split, not random split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train a k-NN classifier with DTW
knn = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

**Key Points:**

- **Time-Based Split**: Always split chronologically for time series, never randomly
- **DTW Metric**: Uses DTW distance to find nearest neighbors
- **Interpretability**: Easy to understand—classify based on similarity to training examples

**When to Use kNN with DTW:**

- **Small Datasets**: Works well with limited training data
- **Interpretability**: When you need to understand why a classification was made
- **Baseline**: Good baseline for time series classification
- **Variable Patterns**: When patterns have variable timing

**Limitations:**

- **Computational Cost**: Slow for large datasets (must compute DTW for all pairs)
- **Memory**: Stores all training examples
- **Sensitivity to k**: Choice of k (number of neighbors) affects performance

---

## 19.6 Feature Extraction: Symbolic Aggregate Approximation (SAX)

SAX converts time series into symbolic representations, making them easier to work with for some algorithms.

### Understanding SAX

SAX works by:

1. **Normalizing**: Standardize the time series
2. **Piecewise Aggregation**: Divide into segments and compute averages
3. **Symbolic Representation**: Convert averages to symbols (letters)

### SAX Example

```python
from tslearn.piecewise import SymbolicAggregateApproximation

# Apply SAX
sax = SymbolicAggregateApproximation(n_segments=5, alphabet_size_avg=3)
X_sax = sax.fit_transform(X)

print(X_sax)
```

**Benefits of SAX:**

- **Dimensionality Reduction**: Reduces time series to a small number of symbols
- **Noise Reduction**: Aggregation smooths out noise
- **Interpretability**: Symbolic representation is human-readable
- **Efficiency**: Faster to process than raw time series

**Use Cases:**

- **Pattern Mining**: Finding recurring patterns in symbolic form
- **Indexing**: Creating indexes for fast similarity search
- **Visualization**: Visualizing time series patterns
- **Feature Engineering**: Creating features for machine learning

---

## 19.7 Practical Applications

### Application 1: Anomaly Detection

Use clustering to identify anomalous time series:

```python
# Cluster time series
clusters = kshape.fit_predict(X)

# Find series that don't fit well into any cluster
# (high distance to nearest centroid indicates anomaly)
distances = []
for i, series in enumerate(X):
    cluster_id = clusters[i]
    centroid = kshape.cluster_centers_[cluster_id]
    distance = dtw(series.ravel(), centroid.ravel())
    distances.append(distance)

# Identify anomalies (e.g., top 5% by distance)
threshold = np.percentile(distances, 95)
anomalies = np.where(distances > threshold)[0]

print(f"Found {len(anomalies)} anomalous time series")
```

### Application 2: Pattern Discovery

Use clustering to discover common patterns:

```python
# Cluster and visualize patterns
for cluster_id in range(kshape.n_clusters):
    cluster_series = X[clusters == cluster_id]
    centroid = kshape.cluster_centers_[cluster_id]
    
    plt.figure(figsize=(10, 4))
    for series in cluster_series[:10]:  # Show first 10 in cluster
        plt.plot(series.ravel(), alpha=0.3, color='gray')
    plt.plot(centroid.ravel(), 'r-', linewidth=2, label='Centroid')
    plt.title(f'Cluster {cluster_id} Pattern')
    plt.legend()
    plt.show()
```

### Application 3: Similarity Search

Find time series similar to a query:

```python
# Find most similar time series to a query
query_series = X[0]

# Compute DTW distance to all other series
distances = [dtw(query_series.ravel(), series.ravel()) for series in X[1:]]

# Find k most similar
k = 5
most_similar_indices = np.argsort(distances)[:k]

print(f"Most similar time series indices: {most_similar_indices}")
```

---

## 19.8 Best Practices

### 1. Preprocessing

- **Normalize**: Always normalize before clustering or classification
- **Resample**: Ensure consistent lengths when needed
- **Handle Missing Values**: Impute or remove missing values appropriately

### 2. Distance Metrics

- **DTW**: Use for variable-speed or phase-shifted patterns
- **Euclidean**: Use for aligned, same-length series
- **Shape-Based**: Use for normalized shape patterns

### 3. Clustering

- **Choose k Carefully**: Use domain knowledge or elbow method
- **Normalize First**: Essential for meaningful clusters
- **Visualize Centroids**: Understand what each cluster represents

### 4. Classification

- **Time-Based Split**: Never use random splits for time series
- **Cross-Validation**: Use time series cross-validation (e.g., TimeSeriesSplit)
- **Feature Engineering**: Consider SAX or other feature extraction methods

### 5. Computational Efficiency

- **DTW Constraints**: Use window constraints to speed up DTW
- **Sampling**: Downsample long time series when appropriate
- **Parallel Processing**: Use parallel computation for large datasets

---

## 19.9 Summary

This chapter introduced key concepts for time series similarity and clustering:

**Key Concepts:**

1. **Dynamic Time Warping (DTW)**: Measures similarity allowing for time warping
2. **K-Shape Clustering**: Shape-based clustering for time series
3. **DTW K-Means**: K-Means with DTW distance metric
4. **kNN Classification**: Classification using DTW-based nearest neighbors
5. **SAX**: Symbolic representation for feature extraction

**When to Use Each Method:**

- **DTW**: Variable-speed patterns, phase shifts, different lengths
- **K-Shape**: Normalized shape patterns, interpretable clusters
- **DTW K-Means**: Variable-speed patterns with K-Means approach
- **kNN with DTW**: Classification with interpretable results
- **SAX**: Dimensionality reduction, pattern mining, indexing

**Best Practices:**

- Always normalize before clustering
- Use time-based splits for classification
- Visualize clusters to understand patterns
- Consider computational cost when choosing methods

---

## Exercises

1. **DTW Distance**: Compute DTW distances between multiple time series. Visualize the warping paths.

2. **Clustering Comparison**: Compare K-Shape and DTW K-Means on the same dataset. Which produces more interpretable clusters?

3. **Classification**: Build a kNN classifier with DTW for a time series classification problem. Experiment with different values of k.

4. **SAX Analysis**: Apply SAX to a time series dataset. Analyze how different parameters (n_segments, alphabet_size) affect the representation.

5. **Anomaly Detection**: Use clustering to detect anomalies in a time series dataset. Evaluate the results.

---

## References and Further Reading

- tslearn Documentation: https://tslearn.readthedocs.io/
- Dynamic Time Warping: Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition.
- K-Shape: Paparrizos, J., & Gravano, L. (2015). k-Shape: Efficient and accurate clustering of time series.
- SAX: Lin, J., Keogh, E., Wei, L., & Lonardi, S. (2007). Experiencing SAX: a novel symbolic representation of time series.

