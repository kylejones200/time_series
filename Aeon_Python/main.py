#!/usr/bin/env python3
"""Aeon clustering template using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from aeon.clustering import TimeSeriesKMeans
from aeon.distances import dtw_distance
from sklearn.metrics import silhouette_score


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    n_clusters: int
    distance: str
    season_length: int
    colors: List[str]
    output_dir: Path
    output_plot: Path
    figsize: Tuple[int, int]
    dpi: int


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    repo_root = script_dir.parent
    data_cfg = config_dict["data"]
    cluster_cfg = config_dict["clustering"]
    plot_cfg = config_dict["plotting"]
    output_dir = ensure_output_dir(Path(script_dir) / config_dict["output"]["output_dir"])
    
    figsize = tuple(plot_cfg.get("figsize", [16, 10]))
    dpi = int(plot_cfg.get("dpi", 150))
    
    return Config(
        data_path=repo_root / "data" / data_cfg["input_file"],
        date_col=data_cfg["date_col"],
        value_col=data_cfg["value_col"],
        freq=data_cfg.get("freq", "MS"),
        n_clusters=int(cluster_cfg.get("n_clusters", 3)),
        distance=cluster_cfg.get("distance", "dtw"),
        season_length=int(cluster_cfg.get("season_length", 12)),
        colors=plot_cfg.get(
            "colors",
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"],
        ),
        output_dir=output_dir,
        output_plot=output_dir / config_dict["output"]["cluster_summary_plot"],
        figsize=figsize,
        dpi=dpi,
    )


def load_series(config: Config) -> pd.Series:
    """Load time series using consolidated loader."""
    from src import load_time_series
    series = load_time_series(
        str(config.data_path),
        date_column=config.date_col,
        value_column=config.value_col
    )
    
    if config.freq:
        series = series.asfreq(config.freq)
    
    return series.astype(float)


def prepare_annual_sequences(
    series: pd.Series, season_length: int
) -> Tuple[np.ndarray, List[int]]:
    """Prepare annual sequences for clustering."""
    df = series.to_frame("value")
    df["year"] = df.index.year
    df["month"] = df.index.month
    
    year_counts = df.groupby("year").size()
    complete_years = year_counts[year_counts == season_length].index
    df_complete = df[df["year"].isin(complete_years)]
    
    sequences = []
    years = []
    for year in complete_years:
        year_data = df_complete[df_complete["year"] == year]["value"].values
        if len(year_data) == season_length:
            sequences.append(year_data)
            years.append(year)
    
    return np.array(sequences), years


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Load series
    series = load_series(config)
    print(f"Loaded {len(series)} data points")
    
    # Prepare sequences
    print("\nPreparing annual sequences for clustering...")
    sequences, years = prepare_annual_sequences(series, config.season_length)
    print(f"Prepared {len(sequences)} complete sequences")
    
    # Fit clustering model
    print(f"\nFitting TimeSeriesKMeans model ({config.distance} distance)...")
    model = TimeSeriesKMeans(
        n_clusters=config.n_clusters,
        metric=config.distance,
        random_state=42,
    )
    
    labels = model.fit_predict(sequences)
    
    # Evaluate
    if len(sequences) > 1:
        silhouette = silhouette_score(sequences.reshape(len(sequences), -1), labels)
        print(f"\nClustering Results:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Number of clusters: {config.n_clusters}")
    
    # Create visualization
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=config.figsize)
    
    for cluster_id in range(config.n_clusters):
        cluster_mask = labels == cluster_id
        cluster_sequences = sequences[cluster_mask]
        
        for seq in cluster_sequences:
            ax.plot(seq, color=config.colors[cluster_id % len(config.colors)], alpha=0.3, lw=1)
        
        # Plot cluster centroid
        if len(cluster_sequences) > 0:
            centroid = np.mean(cluster_sequences, axis=0)
            ax.plot(centroid, color=config.colors[cluster_id % len(config.colors)], lw=2, label=f"Cluster {cluster_id + 1} (n={cluster_mask.sum()})")
    
    ax.set_title(f"Aeon TimeSeriesKMeans Clustering ({config.distance} distance)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, config.output_plot, dpi=config.dpi)
    plt.close(fig)
    print(f" Clustering plot saved -> {config.output_plot}")
    
    print("\n Aeon clustering complete")


if __name__ == "__main__":
    main()
