#!/usr/bin/env python3
"""Irregular time series resampling and interpolation using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataclasses import dataclass
from typing import Optional

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
from src.config import parse_common_config

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


@dataclass
class Config:
    """Configuration dataclass for this template."""
    resample_rule: str
    start: str
    freq: str
    n_points: int
    gap_prob: float
    gp_length_scale: float
    output_dir: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    common = parse_common_config(config_dict, script_dir)

    sim_cfg = config_dict["simulation"]
    gp_cfg = config_dict["gaussian_process"]
    
    return Config(
        resample_rule=config_dict["resample"]["rule"],
        start=sim_cfg["start"],
        freq=sim_cfg["freq"],
        n_points=sim_cfg["n_points"],
        gap_prob=sim_cfg["gap_probability"],
        gp_length_scale=gp_cfg["length_scale"],
        output_dir=common.output_dir,
    )


def simulate_irregular_series(config: Config) -> pd.Series:
    """Simulate irregular time series with gaps."""
    rng = np.random.default_rng(42)
    full_index = pd.date_range(config.start, periods=config.n_points, freq=config.freq)
    
    mask = rng.random(config.n_points) > config.gap_prob
    observed_index = full_index[mask]
    values = np.cumsum(rng.normal(loc=0.0, scale=1.0, size=mask.sum())) + 10
    
    series = pd.Series(values, index=observed_index, name="value")
    return series


def resample_series(series: pd.Series, rule: str) -> pd.Series:
    """Resample series using forward fill."""
    return series.resample(rule).ffill()


def interpolate_gp(series: pd.Series, config: Config) -> pd.Series:
    """Interpolate missing values using Gaussian Process."""
    # Convert time index to numeric
    X = np.array([(t - series.index[0]).days for t in series.index]).reshape(-1, 1)
    y = series.values
    
    # Fit GP
    kernel = RBF(length_scale=config.gp_length_scale)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gp.fit(X, y)
    
    # Predict on full range
    full_index = pd.date_range(series.index[0], series.index[-1], freq=config.freq)
    X_full = np.array([(t - series.index[0]).days for t in full_index]).reshape(-1, 1)
    y_pred, y_std = gp.predict(X_full, return_std=True)
    
    return pd.Series(y_pred, index=full_index, name="gp_interpolated"), y_std


def plot_interpolation_comparison(
    original: pd.Series,
    resampled: pd.Series,
    gp_interpolated: pd.Series,
    gp_std: np.ndarray,
    config: Config
) -> None:
    """Plot comparison of interpolation methods."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Original (irregular)
    axes[0].scatter(original.index, original.values, s=20, alpha=0.6, color="k", label="Original (irregular)")
    axes[0].set_title("Original Irregular Series")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Resampled (forward fill)
    axes[1].plot(resampled.index, resampled.values, "b-", linewidth=1.5, label="Resampled (forward fill)", alpha=0.8)
    axes[1].scatter(original.index, original.values, s=20, alpha=0.6, color="k", label="Original")
    axes[1].set_title("Resampled (Forward Fill)")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # GP interpolated
    axes[2].plot(gp_interpolated.index, gp_interpolated.values, "r-", linewidth=1.5, label="GP Interpolated", alpha=0.8)
    axes[2].fill_between(
        gp_interpolated.index,
        gp_interpolated.values - 2 * gp_std,
        gp_interpolated.values + 2 * gp_std,
        alpha=0.2,
        color="r",
        label="95% CI"
    )
    axes[2].scatter(original.index, original.values, s=20, alpha=0.6, color="k", label="Original")
    axes[2].set_title("Gaussian Process Interpolation")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Value")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, config.output_dir / "irregular_series_interpolation.png", dpi=300)
    plt.close(fig)
    print(f" Interpolation plot saved -> {config.output_dir / 'irregular_series_interpolation.png'}")


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Simulate irregular series
    print("Simulating irregular time series...")
    original = simulate_irregular_series(config)
    print(f"Generated {len(original)} observed points (out of {config.n_points} possible)")
    
    # Resample using forward fill
    print(f"\nResampling with rule: {config.resample_rule}")
    resampled = resample_series(original, config.resample_rule)
    print(f"Resampled to {len(resampled)} points")
    
    # GP interpolation
    print("\nInterpolating with Gaussian Process...")
    gp_interpolated, gp_std = interpolate_gp(original, config)
    print(f"GP interpolated to {len(gp_interpolated)} points")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_interpolation_comparison(original, resampled, gp_interpolated, gp_std, config)
    
    print("\n Irregular series analysis complete")
    
    if config_dict.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
