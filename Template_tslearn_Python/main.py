#!/usr/bin/env python3
"""
tslearn for Time Series Machine Learning
Machine learning algorithms specifically designed for time series data.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col=config['data']['date_col'])
    
    data = df[config['data']['value_col']].values.reshape(-1, 1)
    
    scaler = TimeSeriesScalerMeanVariance()
    data_scaled = scaler.fit_transform(to_time_series_dataset(data))
    
    model = TimeSeriesKMeans(
        n_clusters=config['model']['n_clusters'],
        metric=config['model']['metric'],
        max_iter=config['model']['max_iter'],
        random_state=42
    )
    
    labels = model.fit_predict(data_scaled)
    
    silhouette = silhouette_score(data_scaled.reshape(len(data_scaled), -1), labels)
    print(f"\nClustering Results:")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Number of clusters: {config['model']['n_clusters']}")
    
    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, {'plotting': config['plotting']})
    
    colors = plt.cm.tab10(np.linspace(0, 1, config['model']['n_clusters']))
    
    for cluster_id in range(config['model']['n_clusters']):
        cluster_mask = labels == cluster_id
        cluster_data = data[cluster_mask.flatten()]
        cluster_dates = df.index[cluster_mask.flatten()]
        
        ax.plot(cluster_dates, cluster_data,
                'o', color=colors[cluster_id],
                markersize=config['plotting']['markersize'],
                alpha=config['plotting']['alpha'],
                label=f'Cluster {cluster_id + 1}')
    
    ax.set_title(config['plot_titles']['tslearn_clustering'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config['plotting']['legend'])
    
    output_path = Path(__file__).parent / "outputs" / "tslearn_clustering.png"
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()

