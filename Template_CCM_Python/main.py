#!/usr/bin/env python3
"""
Convergent Cross Mapping (CCM)
Causal inference method for detecting causality in time series using state space reconstruction.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

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


def time_delay_embedding(series, delay, dimension):
    """Reconstruct state space using time-delay embedding."""
    n = len(series)
    embedded = np.zeros((n - (dimension - 1) * delay, dimension))
    for i in range(dimension):
        embedded[:, i] = series[i * delay : n - (dimension - 1 - i) * delay]
    return embedded


def cross_map(source, target, delay, dimension, n_neighbors=None):
    """Cross-map from source to target using state space reconstruction."""
    if n_neighbors is None:
        n_neighbors = dimension + 1

    embedded_target = time_delay_embedding(target, delay, dimension)
    predictions = []

    for i in range(len(embedded_target)):
        distances = np.linalg.norm(embedded_target - embedded_target[i], axis=1)
        neighbors = np.argsort(distances)[1 : n_neighbors + 1]

        weights = 1 / (distances[neighbors] + 1e-10)
        weights /= np.sum(weights)

        prediction = np.sum(weights * source[neighbors])
        predictions.append(prediction)

    return np.array(predictions)


def compute_ccm_correlation(source, target, delay, dimension, n_neighbors=None):
    """Compute CCM correlation between source and target."""
    predictions = cross_map(source, target, delay, dimension, n_neighbors)

    min_len = min(len(source) - (dimension - 1) * delay, len(predictions))
    source_aligned = source[(dimension - 1) * delay : (dimension - 1) * delay + min_len]
    predictions_aligned = predictions[:min_len]

    if len(source_aligned) < 2:
        return 0.0

    correlation = np.corrcoef(source_aligned, predictions_aligned)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def detect_causality(series1, series2, config):
    """Detect bidirectional causality between two time series."""
    delay = config["model"]["delay"]
    dimension = config["model"]["dimension"]
    n_neighbors = config["model"].get("n_neighbors", dimension + 1)

    corr_1_to_2 = compute_ccm_correlation(
        series1, series2, delay, dimension, n_neighbors
    )
    corr_2_to_1 = compute_ccm_correlation(
        series2, series1, delay, dimension, n_neighbors
    )

    return {
        "series1_to_series2": corr_1_to_2,
        "series2_to_series1": corr_2_to_1,
        "bidirectional": abs(corr_1_to_2 - corr_2_to_1)
        < config["model"].get("causality_threshold", 0.1),
    }


def create_visualizations(
    series1, series2, series1_name, series2_name, results, config
):
    """Generate CCM visualizations."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    delay = config["model"]["delay"]
    dimension = config["model"]["dimension"]
    n_neighbors = config["model"].get("n_neighbors", dimension + 1)

    predictions_1_to_2 = cross_map(series1, series2, delay, dimension, n_neighbors)
    predictions_2_to_1 = cross_map(series2, series1, delay, dimension, n_neighbors)

    min_len_1 = min(len(series1) - (dimension - 1) * delay, len(predictions_1_to_2))
    min_len_2 = min(len(series2) - (dimension - 1) * delay, len(predictions_2_to_1))

    series1_aligned = series1[
        (dimension - 1) * delay : (dimension - 1) * delay + min_len_1
    ]
    series2_aligned = series2[
        (dimension - 1) * delay : (dimension - 1) * delay + min_len_2
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for ax in axes.flatten():
        
    axes[0, 0].plot(
        series1_aligned,
        "k-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label=series1_name,
    )
    axes[0, 0].plot(
        predictions_2_to_1[:min_len_1],
        "r--",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label=f"Predicted from {series2_name}",
    )
    axes[0, 0].set_title(
        f'CCM: {series2_name} → {series1_name}\n(Correlation: {results["series2_to_series1"]:.3f})'
    )
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Value")
    ax.legend()

    axes[0, 1].plot(
        series2_aligned,
        "k-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label=series2_name,
    )
    axes[0, 1].plot(
        predictions_1_to_2[:min_len_2],
        "r--",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label=f"Predicted from {series1_name}",
    )
    axes[0, 1].set_title(
        f'CCM: {series1_name} → {series2_name}\n(Correlation: {results["series1_to_series2"]:.3f})'
    )
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Value")
    ax.legend()

    axes[1, 0].scatter(
        series1_aligned,
        predictions_2_to_1[:min_len_1],
        alpha=0.6,
        s=20,
        edgecolors="black",
        linewidths=0.5,
    )
    axes[1, 0].plot(
        [series1_aligned.min(), series1_aligned.max()],
        [series1_aligned.min(), series1_aligned.max()],
        "r--",
        linewidth=config["plotting"]["linewidth"],
    )
    axes[1, 0].set_title(f"{series2_name} → {series1_name} Scatter")
    axes[1, 0].set_xlabel(f"Actual {series1_name}")
    axes[1, 0].set_ylabel(f"Predicted {series1_name}")

    axes[1, 1].scatter(
        series2_aligned,
        predictions_1_to_2[:min_len_2],
        alpha=0.6,
        s=20,
        edgecolors="black",
        linewidths=0.5,
    )
    axes[1, 1].plot(
        [series2_aligned.min(), series2_aligned.max()],
        [series2_aligned.min(), series2_aligned.max()],
        "r--",
        linewidth=config["plotting"]["linewidth"],
    )
    axes[1, 1].set_title(f"{series1_name} → {series2_name} Scatter")
    axes[1, 1].set_xlabel(f"Actual {series2_name}")
    axes[1, 1].set_ylabel(f"Predicted {series2_name}")

    plt.tight_layout()

    output_path = output_dir / "ccm_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def main():
    """Main execution function."""
    config = load_config()

    df1 = load_ts_data(
        data_path=Path(__file__).parent.parent
        / "data"
        / config["data"]["series1_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["series1_col"],
    )
    df1 = ensure_datetime_index(df1, time_col=config["data"]["date_col"])

    df2 = load_ts_data(
        data_path=Path(__file__).parent.parent
        / "data"
        / config["data"]["series2_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["series2_col"],
    )
    df2 = ensure_datetime_index(df2, time_col=config["data"]["date_col"])

    series1 = df1[config["data"]["series1_col"]].values
    series2 = df2[config["data"]["series2_col"]].values

    if config["model"]["normalize"]:
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        series1 = scaler1.fit_transform(series1.reshape(-1, 1)).flatten()
        series2 = scaler2.fit_transform(series2.reshape(-1, 1)).flatten()

    results = detect_causality(series1, series2, config)

    print("\nConvergent Cross Mapping (CCM) Results:")
    print("=" * 70)
    print(
        f"{config['data']['series1_name']} → {config['data']['series2_name']}: {results['series1_to_series2']:.4f}"
    )
    print(
        f"{config['data']['series2_name']} → {config['data']['series1_name']}: {results['series2_to_series1']:.4f}"
    )
    print(f"\nBidirectional causality: {'Yes' if results['bidirectional'] else 'No'}")

    if results["series1_to_series2"] > results["series2_to_series1"]:
        print(
            f"\nStronger direction: {config['data']['series1_name']} → {config['data']['series2_name']}"
        )
    elif results["series2_to_series1"] > results["series1_to_series2"]:
        print(
            f"\nStronger direction: {config['data']['series2_name']} → {config['data']['series1_name']}"
        )
    else:
        print(
            "\nBidirectional causality detected (similar correlation in both directions)"
        )

    create_visualizations(
        series1,
        series2,
        config["data"]["series1_name"],
        config["data"]["series2_name"],
        results,
        config,
    )

    print("✓ CCM analysis complete")


if __name__ == "__main__":
    main()
