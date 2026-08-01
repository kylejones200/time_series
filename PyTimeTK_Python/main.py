#!/usr/bin/env python3
"""PyTimeTK and overview visualisations using consolidated utilities."""

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


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    output_dir: Path
    pytimetk_plot: Path
    overview_plot: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    common = parse_common_config(config_dict, script_dir)

    
    return Config(
        data_path=common.data_path,
        date_col=common.date_col,
        value_col=common.value_col,
        freq=config_dict["data"].get("freq", "MS"),
        output_dir=common.output_dir,
        pytimetk_plot=common.output_dir / config_dict["output"]["pytimetk_plot"],
        overview_plot=common.output_dir / config_dict["output"]["overview_plot"],
    )


def load_dataframe(config: Config) -> pd.DataFrame:
    """Load DataFrame from CSV."""
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")
    
    df = pd.read_csv(config.data_path, encoding="utf-8")
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not present in CSV")
    
    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df[config.value_col] = pd.to_numeric(df[config.value_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.value_col]).sort_values(config.date_col)
    return df


def plot_pytimetk_view(df: pd.DataFrame, config: Config) -> None:
    """Plot PyTimeTK-style view with YoY changes."""
    df = df.copy()
    df["yoy_pct"] = df[config.value_col].pct_change(12) * 100.0
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(df[config.date_col], df[config.value_col], label="Monthly generation")
    axes[0].legend()
    axes[1].plot(
        df[config.date_col], df["yoy_pct"], color="tab:orange", label="YoY % change"
    )
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].legend()
    
    fig.tight_layout()
    save_plot(fig, config.pytimetk_plot, dpi=300)
    plt.close(fig)
    print(f" PyTimeTK plot saved -> {config.pytimetk_plot}")


def plot_overview(df: pd.DataFrame, config: Config) -> None:
    """Plot overview visualization."""
    df = df.copy()
    df["yoy_pct"] = df[config.value_col].pct_change(12) * 100.0
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    axes[0].plot(df[config.date_col], df[config.value_col], "k-", linewidth=1.5, label="Monthly generation")
    axes[0].set_ylabel("Generation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df[config.date_col], df["yoy_pct"], color="tab:orange", linewidth=1.5, label="YoY % change")
    axes[1].axhline(0, color="k", lw=0.5, linestyle="--")
    axes[1].set_ylabel("YoY % Change")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, config.overview_plot, dpi=300)
    plt.close(fig)
    print(f" Overview plot saved -> {config.overview_plot}")


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Load DataFrame
    df = load_dataframe(config)
    print(f"Loaded {len(df)} data points")
    
    # Create visualizations
    plot_pytimetk_view(df, config)
    plot_overview(df, config)
    
    print("\n PyTimeTK analysis complete")


if __name__ == "__main__":
    main()
