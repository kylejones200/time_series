#!/usr/bin/env python3
"""PyTimeTK and overview visualisations aligned with the 2025-11-08 article assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    output_dir: Path
    pytimetk_plot: Path
    overview_plot: Path


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
        output_dir=output_dir,
        pytimetk_plot=output_dir / cfg["output"]["pytimetk_plot"],
        overview_plot=output_dir / cfg["output"]["overview_plot"],
    )


def load_dataframe(config: Config) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")

    df = pd.read_csv(config.data_path)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not present in CSV")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df[config.value_col] = pd.to_numeric(df[config.value_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.value_col]).sort_values(config.date_col)
    return df


def plot_pytimetk_view(df: pd.DataFrame, config: Config) -> None:
    df = df.copy()
    df["yoy_pct"] = df[config.value_col].pct_change(12) * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(df[config.date_col], df[config.value_col], label="Monthly generation")
    axes[0].legend()
    axes[1].plot(df[config.date_col], df["yoy_pct"], color="tab:orange", label="YoY % change")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(config.pytimetk_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ PyTimeTK-inspired plot saved -> {config.pytimetk_plot}")


def plot_overview(df: pd.DataFrame, config: Config) -> None:
    df = df.copy()
    df_monthly = df.set_index(config.date_col).asfreq(config.freq)
    df_yearly = df_monthly.resample("Y").mean()
    df["yoy_pct"] = df[config.value_col].pct_change(12) * 100.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    axes[0].plot(df_monthly.index, df_monthly[config.value_col], label="Monthly")
    axes[0].legend()
    axes[1].bar(df_yearly.index.year, df_yearly[config.value_col], label="Yearly mean")
    axes[1].legend()
    axes[2].plot(df[config.date_col], df["yoy_pct"], color="tab:orange", label="YoY %")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(config.overview_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Overview plot saved -> {config.overview_plot}")


def main() -> None:
    config = load_config()
    df = load_dataframe(config)
    plot_pytimetk_view(df, config)
    plot_overview(df, config)


if __name__ == "__main__":
    main()

