#!/usr/bin/env python3
"""Seasonal naive error diagnostics using consolidated utilities."""

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

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    horizon: int
    n_splits: int
    season: int
    output_dir: Path
    error_plot: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    repo_root = script_dir.parent
    data_path = repo_root / "data" / config_dict["data"]["input_file"]
    output_dir = ensure_output_dir(Path(script_dir) / config_dict["output"]["output_dir"])
    
    return Config(
        data_path=data_path,
        date_col=config_dict["data"]["date_col"],
        value_col=config_dict["data"]["value_col"],
        freq=config_dict["data"].get("freq", "MS"),
        horizon=int(config_dict["evaluation"]["horizon"]),
        n_splits=int(config_dict["evaluation"]["n_splits"]),
        season=int(config_dict["evaluation"]["season"]),
        output_dir=output_dir,
        error_plot=output_dir / config_dict["output"]["error_plot"],
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


def mase_denominator(train: pd.Series, season: int) -> float:
    """Calculate MASE denominator."""
    diffs = np.abs(train.values[season:] - train.values[:-season])
    if len(diffs) == 0 or np.allclose(diffs, 0.0):
        return 1.0
    return float(np.mean(diffs))


def seasonal_naive_forecast(train: pd.Series, horizon: int, season: int) -> np.ndarray:
    """Generate seasonal naive forecast."""
    forecast = []
    values = train.values
    for i in range(horizon):
        src_idx = len(values) - season + i
        if src_idx >= 0:
            forecast.append(values[src_idx])
        else:
            forecast.append(values[-1])
    return np.asarray(forecast, dtype=float)


def rolling_origin_metrics(
    series: pd.Series, config: Config
) -> Tuple[List[dict], pd.Series, pd.Series]:
    """Rolling origin evaluation with error metrics."""
    idx = np.arange(len(series))
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    metrics: List[dict] = []
    last_truth = None
    last_forecast = None
    
    for train_idx, _ in splitter.split(idx):
        end_idx = train_idx[-1]
        train_series = series.iloc[: end_idx + 1]
        future_series = series.iloc[end_idx + 1 : end_idx + 1 + config.horizon]
        
        if future_series.empty:
            continue
        
        forecast_values = seasonal_naive_forecast(train_series, len(future_series), config.season)
        forecast = pd.Series(forecast_values, index=future_series.index)
        
        mae_val = mean_absolute_error(future_series.values, forecast.values)
        mase_denom = mase_denominator(train_series, config.season)
        mase_val = mae_val / mase_denom if mase_denom > 0 else float("inf")
        
        metrics.append({
            "MAE": mae_val,
            "MASE": mase_val,
            "n_points": len(future_series),
        })
        
        last_truth = future_series
        last_forecast = forecast
    
    return metrics, last_truth, last_forecast


def plot_error_analysis(series: pd.Series, metrics: List[dict], last_true: pd.Series, last_forecast: pd.Series, config: Config) -> None:
    """Plot error analysis."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Forecast plot
    axes[0].plot(series.index[-100:], series.values[-100:], "k-", lw=1.5, label="History", alpha=0.8)
    if last_true is not None:
        axes[0].plot(last_true.index, last_true.values, "b-", lw=1.8, label="Actual", alpha=0.8)
    if last_forecast is not None:
        axes[0].plot(last_forecast.index, last_forecast.values, "r--", lw=2.0, label="Seasonal Naive Forecast", alpha=0.8)
    
    axes[0].set_ylabel("Value")
    axes[0].set_title("Seasonal Naive Forecast")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    
    # Error metrics over folds
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        axes[1].plot(range(len(metrics_df)), metrics_df["MAE"], "o-", lw=1.5, label="MAE", alpha=0.8)
        axes[1].plot(range(len(metrics_df)), metrics_df["MASE"], "s-", lw=1.5, label="MASE", alpha=0.8)
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("Error Metric")
        axes[1].set_title("Error Metrics Across Folds")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, config.error_plot, dpi=300)
    plt.close(fig)
    print(f" Error analysis plot saved -> {config.error_plot}")


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
    
    # Rolling origin evaluation
    metrics, last_true, last_forecast = rolling_origin_metrics(series, config)
    
    if metrics:
        mean_mae = np.mean([m["MAE"] for m in metrics])
        mean_mase = np.mean([m["MASE"] for m in metrics])
        print(f"\nRolling Origin Evaluation Results:")
        print(f"  Mean MAE: {mean_mae:.4f}")
        print(f"  Mean MASE: {mean_mase:.4f}")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_error_analysis(series, metrics, last_true, last_forecast, config)
    
    print("\n Forecast error analysis complete")


if __name__ == "__main__":
    main()
