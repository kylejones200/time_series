#!/usr/bin/env python3
"""Seasonal decomposition visuals using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.config import parse_common_config

from statsmodels.tsa.seasonal import seasonal_decompose


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    period: int
    output_dir: Path
    decomposition_plot: Path
    seasonal_plot: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    common = parse_common_config(config_dict, script_dir)

    
    return Config(
        data_path=common.data_path,
        date_col=common.date_col,
        value_col=common.value_col,
        freq=config_dict["data"].get("freq", "MS"),
        period=int(config_dict["model"]["period"]),
        output_dir=common.output_dir,
        decomposition_plot=common.output_dir / config_dict["output"]["decomposition_plot"],
        seasonal_plot=common.output_dir / config_dict["output"]["seasonal_plot"],
    )


def load_series(config: Config) -> pd.Series:
    """Load time series using consolidated loader."""
    # Use consolidated loader
    series = load_time_series(
        str(config.data_path),
        date_column=config.date_col,
        value_column=config.value_col
    )
    
    # Apply frequency conversion if needed
    if config.freq:
        series = series.asfreq(config.freq)
    
    return series.astype(float)


def plot_decomposition(series: pd.Series, config: Config) -> None:
    """Plot seasonal decomposition."""
    decomposition = seasonal_decompose(series, model="additive", period=config.period)
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(series.index, series.values, "k-", linewidth=1.5)
    axes[0].set_title("Observed")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(decomposition.trend.index, decomposition.trend.values, "b-", linewidth=1.5)
    axes[1].set_title("Trend")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, "g-", linewidth=1.5)
    axes[2].set_title("Seasonal")
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(decomposition.resid.index, decomposition.resid.values, "r-", linewidth=1.5)
    axes[3].set_title("Residual")
    axes[3].set_xlabel("Date")
    axes[3].grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, config.decomposition_plot, dpi=300)
    plt.close(fig)
    print(f" Decomposition plot saved -> {config.decomposition_plot}")


def plot_seasonal_pattern(series: pd.Series, config: Config) -> None:
    """Plot seasonal pattern visualization."""
    decomposition = seasonal_decompose(series, model="additive", period=config.period)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(decomposition.seasonal.index, decomposition.seasonal.values, "g-", linewidth=1.5)
    ax.set_title("Seasonal Component")
    ax.set_xlabel("Date")
    ax.set_ylabel("Seasonal Effect")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, config.seasonal_plot, dpi=300)
    plt.close(fig)
    print(f" Seasonal plot saved -> {config.seasonal_plot}")


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Load series using consolidated loader
    series = load_series(config)
    print(f"Loaded {len(series)} data points")
    
    # Create visualizations
    plot_decomposition(series, config)
    plot_seasonal_pattern(series, config)
    
    print("\n Time series decomposition analysis complete")


if __name__ == "__main__":
    main()
