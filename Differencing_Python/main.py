#!/usr/bin/env python3
"""
Differencing for Stationarity
Visualize original series and successive differences (+ADF diagnostics).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from statsmodels.tsa.stattools import adfuller


def fetch_remote_csv(url: str) -> pd.DataFrame:
    """Fetch CSV from remote URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def load_series(config: dict) -> pd.Series:
    """Load time series, handling both remote URLs and local files."""
    if config["data"].get("url"):
        df = fetch_remote_csv(config["data"]["url"])
    else:
        # Use consolidated loader for local files
        return load_time_series(
            config["data"]["input_file"],
            date_column=config["data"].get("date_col", "date"),
            value_column=config["data"].get("value_col", "value")
        )
    
    # Process remote data
    df[config["data"]["date_col"]] = pd.to_datetime(
        df[config["data"]["date_col"]], errors="coerce"
    )
    df = df.dropna(subset=[config["data"]["date_col"], config["data"]["value_col"]])
    df = df.sort_values(config["data"]["date_col"])
    df = df.set_index(config["data"]["date_col"])
    
    if config["data"].get("resample_rule"):
        df = df.resample(config["data"]["resample_rule"]).mean().dropna()
    
    series = pd.to_numeric(df[config["data"]["value_col"]], errors="coerce").dropna()
    series.name = config["data"]["value_col"]
    return series


def adf_summary(series: pd.Series) -> dict:
    """Calculate ADF test summary."""
    result = adfuller(series.dropna(), autolag="AIC")
    keys = ["ADF Statistic", "p-value", "Lags Used", "Number of Observations"]
    return {k: v for k, v in zip(keys, result[:4])}


def compute_differences(series: pd.Series, max_order: int) -> list[pd.Series]:
    """Compute successive differences of the series."""
    diffs = [series]
    current = series
    for _ in range(max_order):
        current = current.diff().dropna()
        diffs.append(current)
    return diffs


def plot_differences(
    series_list: list[pd.Series], config: dict, adf_results: list[dict], script_dir: Path
) -> None:
    """Plot differences with ADF results."""
    figure_cfg = config["plotting"]
    fig, axes = plt.subplots(
        len(series_list), 1, figsize=figure_cfg["figure_size"], sharex=True
    )
    if len(series_list) == 1:
        axes = [axes]
    
    for idx, (series, ax) in enumerate(zip(series_list, axes)):
        ax.plot(
            series.index,
            series.values,
            color=figure_cfg["colors"][0],
            linewidth=figure_cfg["linewidth"],
            alpha=figure_cfg["alpha"],
        )
        
        title = config["plot_titles"]["base"] if idx == 0 else f"{idx} order difference"
        info = adf_results[idx]
        ax.set_title(
            f"{title} | ADF: {info['ADF Statistic']:.3f}, p={info['p-value']:.3f}"
        )
        ax.set_ylabel(config["plotting"].get("y_label", "Value"))
    
    axes[-1].set_xlabel(config["plotting"].get("x_label", "Date"))
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / "differencing_plot.png", dpi=300)
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


def main():
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load series
    series = load_series(config)
    print(f"Loaded {len(series)} data points")
    
    max_order = config["model"]["max_difference_order"]
    series_list = compute_differences(series, max_order)
    adf_results = [adf_summary(s) for s in series_list]
    
    # Save ADF results
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    metrics_path = output_dir / "adf_results.csv"
    pd.DataFrame(adf_results).to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"ADF results saved to {metrics_path}")
    
    # Plot differences
    plot_differences(series_list, config, adf_results, script_dir)
    
    print("\n Differencing analysis complete")


if __name__ == "__main__":
    main()
