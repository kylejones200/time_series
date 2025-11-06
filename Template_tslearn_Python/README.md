# tslearn for Time Series Machine Learning

Machine learning algorithms specifically designed for time series data, including clustering and classification.

## Features

- ✅ Time series clustering (K-Means)
- ✅ Dynamic Time Warping (DTW) support
- ✅ Time series scaling and preprocessing
- ✅ Silhouette score evaluation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file in the shared `data/` directory
2. Update `config.yaml` with your data file name and column names
3. Run:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- **n_clusters**: Number of clusters
- **metric**: Distance metric (`euclidean`, `dtw`, `softdtw`)
- **max_iter**: Maximum iterations for clustering

## Metrics

- **euclidean**: Standard Euclidean distance
- **dtw**: Dynamic Time Warping (handles time shifts)
- **softdtw**: Soft DTW (differentiable)

## Applications

- Time series clustering
- Pattern discovery
- Anomaly detection
- Time series classification

## Outputs

- `outputs/tslearn_clustering.png`: Clustering visualization
- Console output: Silhouette score and cluster statistics

