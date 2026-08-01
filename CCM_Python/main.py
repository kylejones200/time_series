#!/usr/bin/env python3
"""
Convergent Cross Mapping (CCM)
Causal inference method for detecting causality in time series using state space reconstruction.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def time_delay_embedding(series: np.ndarray, delay: int, dimension: int):
    """Reconstruct state space using time-delay embedding."""
    n = len(series)
    embedded = np.zeros((n - (dimension - 1) * delay, dimension))
    for i in range(dimension):
        embedded[:, i] = series[i * delay : n - (dimension - 1 - i) * delay]
    return embedded


def cross_map(source: np.ndarray, target: np.ndarray, delay: int, dimension: int, n_neighbors: int = None):
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


def compute_ccm_correlation(source: np.ndarray, target: np.ndarray, delay: int, dimension: int, n_neighbors: int = None):
    """Compute CCM correlation between source and target."""
    predictions = cross_map(source, target, delay, dimension, n_neighbors)
    
    min_len = min(len(source) - (dimension - 1) * delay, len(predictions))
    source_aligned = source[(dimension - 1) * delay : (dimension - 1) * delay + min_len]
    predictions_aligned = predictions[:min_len]
    
    if len(source_aligned) < 2:
        return 0.0
    
    correlation = np.corrcoef(source_aligned, predictions_aligned)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def detect_causality(series1: np.ndarray, series2: np.ndarray, config: dict):
    """Detect bidirectional causality between two time series."""
    delay = config["model"]["delay"]
    dimension = config["model"]["dimension"]
    n_neighbors = config["model"].get("n_neighbors", dimension + 1)
    
    corr_1_to_2 = compute_ccm_correlation(series1, series2, delay, dimension, n_neighbors)
    corr_2_to_1 = compute_ccm_correlation(series2, series1, delay, dimension, n_neighbors)
    
    return {
        "series1_to_series2": corr_1_to_2,
        "series2_to_series1": corr_2_to_1,
        "bidirectional": abs(corr_1_to_2 - corr_2_to_1) < config["model"].get("causality_threshold", 0.1),
    }


def create_visualizations(series1: np.ndarray, series2: np.ndarray, series1_name: str, series2_name: str, results: dict, config: dict, script_dir: Path):
    """Generate CCM visualizations."""
    delay = config["model"]["delay"]
    dimension = config["model"]["dimension"]
    n_neighbors = config["model"].get("n_neighbors", dimension + 1)
    
    predictions_1_to_2 = cross_map(series1, series2, delay, dimension, n_neighbors)
    predictions_2_to_1 = cross_map(series2, series1, delay, dimension, n_neighbors)
    
    min_len_1 = min(len(series1) - (dimension - 1) * delay, len(predictions_1_to_2))
    min_len_2 = min(len(series2) - (dimension - 1) * delay, len(predictions_2_to_1))
    
    series1_aligned = series1[(dimension - 1) * delay : (dimension - 1) * delay + min_len_1]
    series2_aligned = series2[(dimension - 1) * delay : (dimension - 1) * delay + min_len_2]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(
        series1_aligned,
        "k-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label=series1_name,
    )
    axes[0, 0].plot(
        predictions_2_to_1[:min_len_1],
        "r--",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label=f"Predicted from {series2_name}",
    )
    axes[0, 0].set_title(f"Predicting {series1_name} from {series2_name}")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(
        series2_aligned,
        "k-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label=series2_name,
    )
    axes[0, 1].plot(
        predictions_1_to_2[:min_len_2],
        "b--",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label=f"Predicted from {series1_name}",
    )
    axes[0, 1].set_title(f"Predicting {series2_name} from {series1_name}")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(series1_aligned, predictions_2_to_1[:min_len_1], alpha=0.6, s=20)
    axes[1, 0].plot([series1_aligned.min(), series1_aligned.max()], [series1_aligned.min(), series1_aligned.max()], "r--", lw=2)
    axes[1, 0].set_xlabel(f"Actual {series1_name}")
    axes[1, 0].set_ylabel(f"Predicted {series1_name}")
    axes[1, 0].set_title(f"Correlation: {results['series2_to_series1']:.3f}")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(series2_aligned, predictions_1_to_2[:min_len_2], alpha=0.6, s=20)
    axes[1, 1].plot([series2_aligned.min(), series2_aligned.max()], [series2_aligned.min(), series2_aligned.max()], "b--", lw=2)
    axes[1, 1].set_xlabel(f"Actual {series2_name}")
    axes[1, 1].set_ylabel(f"Predicted {series2_name}")
    axes[1, 1].set_title(f"Correlation: {results['series1_to_series2']:.3f}")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / "ccm_analysis.png", dpi=300)
    print(f"Plot saved to: {output_dir / 'ccm_analysis.png'}")


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data - CCM requires two series
    data_cfg = config["data"]
    if "input_file" in data_cfg:
        data_path = Path(data_cfg["input_file"])
        if not data_path.is_absolute():
            data_path = script_dir.parent / "data" / data_cfg["input_file"]
        df = pd.read_csv(data_path, encoding="utf-8")
        series1_col = data_cfg.get("series1_col", "value")
        series2_col = data_cfg.get("series2_col", "value")
        series1 = df[series1_col].values
        series2 = df[series2_col].values
    else:
        # Fallback for split-file configs
        def _load_series(path_key: str, value_key: str) -> np.ndarray:
            p = Path(data_cfg[path_key])
            if not p.is_absolute():
                p = script_dir.parent / "data" / data_cfg[path_key]
            s = load_time_series(
                str(p),
                date_column=data_cfg.get("date_col", "date"),
                value_column=data_cfg.get(value_key, "value"),
            )
            return s.values

        series1 = _load_series("series1_file", "series1_col")
        series2 = _load_series("series2_file", "series2_col")
    
    print(f"Loaded {len(series1)} data points for both series")
    
    # Detect causality
    print("\nDetecting causality using CCM...")
    results = detect_causality(series1, series2, config)
    
    print(f"\nCCM Results:")
    print(f"  {config['data']['series1_name']} → {config['data']['series2_name']}: {results['series1_to_series2']:.4f}")
    print(f"  {config['data']['series2_name']} → {config['data']['series1_name']}: {results['series2_to_series1']:.4f}")
    print(f"  Bidirectional: {results['bidirectional']}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(
        series1,
        series2,
        config["data"]["series1_name"],
        config["data"]["series2_name"],
        results,
        config,
        script_dir,
    )
    
    print("\n CCM analysis complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
