#!/usr/bin/env python3
"""Aeon clustering template aligned with the 2025-11-08 article assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import signalplot
import numpy as np
import pandas as pd
import yaml
from aeon.clustering import TimeSeriesKMeans
from aeon.distances import dtw_distance
from sklearn.metrics import silhouette_score

# Apply SignalPlot's clean defaults
signalplot.apply()


@dataclass
class Config:
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


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_cfg = cfg["data"]
    cluster_cfg = cfg["clustering"]
    plot_cfg = cfg["plotting"]
    output_dir = Path(__file__).parent / cfg["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

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
        output_plot=output_dir / cfg["output"]["cluster_summary_plot"],
        figsize=figsize,
        dpi=dpi,
    )


def load_series(config: Config) -> pd.Series:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")

    df = pd.read_csv(config.data_path)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not present in CSV")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.value_col])
    df = df.sort_values(config.date_col).set_index(config.date_col)
    series = pd.to_numeric(df[config.value_col], errors="coerce").dropna()
    series = series.asfreq(config.freq)
    return series.astype(float)


def prepare_annual_sequences(
    series: pd.Series, season_length: int
) -> Tuple[np.ndarray, List[int]]:
    df = series.to_frame("value")
    df["year"] = df.index.year
    df["month"] = df.index.month

    year_counts = df.groupby("year").size()
    complete_years = year_counts[year_counts == season_length].index
    df_complete = df[df["year"].isin(complete_years)]

    sequences = []
    years = []
    for year in complete_years:
        year_values = (
            df_complete[df_complete["year"] == year]
            .sort_values("month")["value"]
            .values
        )
        if len(year_values) == season_length:
            sequences.append(year_values)
            years.append(int(year))

    if not sequences:
        raise ValueError("Not enough complete seasonal cycles to cluster.")

    array = np.array(sequences, dtype=float).reshape(len(sequences), 1, season_length)
    return array, years


def compute_dtw_matrix(sequences: np.ndarray) -> np.ndarray:
    n = sequences.shape[0]
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = dtw_distance(sequences[i, 0], sequences[j, 0])
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix


def run_clustering(config: Config, sequences: np.ndarray) -> TimeSeriesKMeans:
    model = TimeSeriesKMeans(
        n_clusters=config.n_clusters,
        distance=config.distance,
        n_init=10,
        random_state=42,
        max_iter=50,
    )
    model.fit(sequences)
    return model


def plot_cluster_summary(
    config: Config,
    years: List[int],
    sequences: np.ndarray,
    labels: np.ndarray,
    model: TimeSeriesKMeans,
    dtw_matrix: np.ndarray,
) -> None:
    months = np.arange(1, sequences.shape[2] + 1)
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    fig = plt.figure(figsize=config.figsize, dpi=config.dpi)

    ax1 = plt.subplot(2, 3, 1)
    for idx, year in enumerate(years):
        color = config.colors[labels[idx] % len(config.colors)]
        ax1.scatter(year, labels[idx], c=color, s=100, alpha=0.7)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cluster")
    ax1.set_title("Cluster assignments over time")
    ax1.set_yticks(range(config.n_clusters))

    ax2 = plt.subplot(2, 3, 2)
    for cluster in range(config.n_clusters):
        centroid = model.cluster_centers_[cluster, 0]
        ax2.plot(
            months,
            centroid,
            label=f"Cluster {cluster}",
            linewidth=2.5,
            marker="o",
            color=config.colors[cluster % len(config.colors)],
        )
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Generation (thousand MWh)")
    ax2.set_title("Seasonal centroids")
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_names, rotation=45, ha="right")
    ax2.legend()

    ax3 = plt.subplot(2, 3, 3)
    for idx in range(len(sequences)):
        color = config.colors[labels[idx] % len(config.colors)]
        ax3.plot(months, sequences[idx, 0, :], color=color, alpha=0.3, linewidth=1)
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Generation (thousand MWh)")
    ax3.set_title("All annual profiles by cluster")
    ax3.set_xticks(months)
    ax3.set_xticklabels(month_names, rotation=45, ha="right")

    ax4 = plt.subplot(2, 3, 4)
    im = ax4.imshow(dtw_matrix, cmap="YlOrRd", aspect="auto")
    ax4.set_xlabel("Year index")
    ax4.set_ylabel("Year index")
    ax4.set_title("DTW distance matrix")
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label="DTW distance")

    ax5 = plt.subplot(2, 3, 5)
    for cluster in range(config.n_clusters):
        indices = np.where(labels == cluster)[0]
        if len(indices) == 0:
            continue
        idx = indices[0]
        color = config.colors[cluster % len(config.colors)]
        ax5.plot(
            months,
            sequences[idx, 0, :],
            label=f"Cluster {cluster} ({years[idx]})",
            linewidth=2,
            marker="o",
            color=color,
        )
    ax5.set_xlabel("Month")
    ax5.set_ylabel("Generation (thousand MWh)")
    ax5.set_title("Representative year per cluster")
    ax5.set_xticks(months)
    ax5.set_xticklabels(month_names, rotation=45, ha="right")
    ax5.legend()

    ax6 = plt.subplot(2, 3, 6)
    flattened = sequences.reshape(sequences.shape[0], -1)
    sil_score = silhouette_score(flattened, labels, metric="euclidean")
    inertia = getattr(model, "inertia_", float("nan"))
    metrics_text = (
        "Clustering metrics\n\n"
        f"Distance: {config.distance.upper()}\n"
        f"Clusters: {config.n_clusters}\n"
        f"Inertia: {inertia:.1f}\n"
        f"Silhouette: {sil_score:.3f}\n"
        f"Iterations: {getattr(model, 'n_iter_', 0)}"
    )
    ax6.text(
        0.05,
        0.5,
        metrics_text,
        transform=ax6.transAxes,
        fontsize=11,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    ax6.axis("off")

    fig.tight_layout()
    fig.savefig(config.output_plot, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Cluster summary plot saved -> {config.output_plot}")


def main() -> None:
    config = load_config()
    series = load_series(config)
    sequences, years = prepare_annual_sequences(series, config.season_length)

    print(f"Total observations: {len(series)}")
    print(f"Complete years used: {len(years)} ({years[0]}–{years[-1]})")

    model = run_clustering(config, sequences)
    labels = model.labels_
    dtw_matrix = compute_dtw_matrix(sequences)

    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        associated_years = [
            years[idx] for idx in range(len(years)) if labels[idx] == cluster
        ]
        print(f"Cluster {cluster}: {count} years ({pct:.1f}%) → {associated_years}")

    plot_cluster_summary(config, years, sequences, labels, model, dtw_matrix)


if __name__ == "__main__":
    main()
