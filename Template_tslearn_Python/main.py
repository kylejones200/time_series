#!/usr/bin/env python3
"""
tslearn for Time Series Machine Learning
Machine learning algorithms specifically designed for time series data.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

# Apply SignalPlot's clean defaults
signalplot.apply()


def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj



ts_utils = repo_import("utils.ts_utils")
load_ts_data = ts_utils.load_ts_data
ensure_datetime_index = ts_utils.ensure_datetime_index


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_ts_data(
        data_path=Path(__file__).parent.parent / "data" / config["data"]["input_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["value_col"],
    )
    df = ensure_datetime_index(df, time_col=config["data"]["date_col"])

    data = df[config["data"]["value_col"]].values.reshape(-1, 1)

    # NOTE: For clustering (unsupervised learning), scaling the entire dataset is acceptable
    # since there's no train/test split. However, if this code is adapted for prediction,
    # the data should be split first, then the scaler should be fit on training data only.
    scaler = TimeSeriesScalerMeanVariance()
    data_scaled = scaler.fit_transform(to_time_series_dataset(data))

    model = TimeSeriesKMeans(
        n_clusters=config["model"]["n_clusters"],
        metric=config["model"]["metric"],
        max_iter=config["model"]["max_iter"],
        random_state=42,
    )

    labels = model.fit_predict(data_scaled)

    silhouette = silhouette_score(data_scaled.reshape(len(data_scaled), -1), labels)
    print(f"\nClustering Results:")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Number of clusters: {config['model']['n_clusters']}")

    fig, ax = plt.subplots(figsize=tuple(config["plotting"]["figure_size"]))
    
    colors = plt.cm.tab10(np.linspace(0, 1, config["model"]["n_clusters"]))

    for cluster_id in range(config["model"]["n_clusters"]):
        cluster_mask = labels == cluster_id
        cluster_data = data[cluster_mask.flatten()]
        cluster_dates = df.index[cluster_mask.flatten()]

        ax.plot(
            cluster_dates,
            cluster_data,
            "o",
            color=colors[cluster_id],
            markersize=config["plotting"]["markersize"],
            alpha=config["plotting"]["alpha"],
            label=f"Cluster {cluster_id + 1}",
        )

    ax.set_title(config["plot_titles"]["tslearn_clustering"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    output_path = Path(__file__).parent / "outputs" / "tslearn_clustering.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
