#!/usr/bin/env python3
"""Seasonal decomposition visuals aligned with the 2025-11-08 article assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import signalplot
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.seasonal import seasonal_decompose

# Apply SignalPlot's clean defaults
signalplot.apply()


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    period: int
    output_dir: Path
    decomposition_plot: Path
    seasonal_plot: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg["data"]["input_file"]
    output_dir = Path(__file__).parent / cfg["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    return Config(
        data_path=data_path,
        date_col=cfg["data"]["date_col"],
        value_col=cfg["data"]["value_col"],
        freq=cfg["data"].get("freq", "MS"),
        period=int(cfg["model"]["period"]),
        output_dir=output_dir,
        decomposition_plot=output_dir / cfg["output"]["decomposition_plot"],
        seasonal_plot=output_dir / cfg["output"]["seasonal_plot"],
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
    return series.asfreq(config.freq).astype(float)


def plot_decomposition(series: pd.Series, config: Config) -> None:
    decomposition = seasonal_decompose(series, model="additive", period=config.period)

    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(series.index, series.values)
    axes[0].set_title("Observed")

    axes[1].plot(decomposition.trend.index, decomposition.trend.values)
    axes[1].set_title("Trend")

    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values)
    axes[2].set_title("Seasonal")

    axes[3].plot(decomposition.resid.index, decomposition.resid.values)
    axes[3].set_title("Residual")

    fig.tight_layout()
    fig.savefig(config.decomposition_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Decomposition plot saved -> {config.decomposition_plot}")


def plot_seasonal_subseries(series: pd.Series, config: Config) -> None:
    df = series.to_frame("value")
    df["month"] = df.index.month
    df["year"] = df.index.year

    fig, ax = plt.subplots(figsize=(10, 6))
    for month in range(1, 13):
        subset = df[df["month"] == month]
        ax.plot(subset["year"], subset["value"], label=f"M{month:02d}", alpha=0.6)

    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.set_title("Seasonal subseries by month")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.seasonal_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Seasonal subseries plot saved -> {config.seasonal_plot}")


def main() -> None:
    config = load_config()
    series = load_series(config)
    plot_decomposition(series, config)
    plot_seasonal_subseries(series, config)


if __name__ == "__main__":
    main()
