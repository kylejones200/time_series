#!/usr/bin/env python3
"""StatsForecast AutoARIMA template using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utilsforecast.losses import mae, mape, mse


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    season_length: int
    prediction_length: int
    holdout_length: int
    output_dir: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    repo_root = script_dir.parent
    data_path = repo_root / "data" / config_dict["data"]["input_file"]
    
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")
    
    output_dir = ensure_output_dir(Path(script_dir) / "outputs")
    
    model_cfg = config_dict["model"]
    return Config(
        data_path=data_path,
        date_col=config_dict["data"]["date_col"],
        value_col=config_dict["data"]["value_col"],
        freq=config_dict["data"].get("freq", "H"),
        season_length=model_cfg.get("season_length", 24),
        prediction_length=model_cfg["prediction_length"],
        holdout_length=model_cfg.get("holdout_length", model_cfg["prediction_length"]),
        output_dir=output_dir,
    )


def load_series(config: Config) -> pd.DataFrame:
    """Load series and prepare for StatsForecast."""
    df = pd.read_csv(config.data_path)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not found in CSV")
    
    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.value_col])
    df = df.sort_values(config.date_col)
    
    df = df.reset_index(drop=True)
    df = df[[config.date_col, config.value_col]].rename(
        columns={config.date_col: "ds", config.value_col: "y"}
    )
    return df


def prepare_data(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare train/test split."""
    df_resampled = df.set_index("ds").resample(config.freq)["y"].mean().reset_index()
    df_resampled["unique_id"] = "series_1"
    
    if config.holdout_length >= len(df_resampled):
        raise ValueError("Holdout length must be smaller than series length")
    
    train = df_resampled.iloc[: -config.holdout_length].copy()
    test = df_resampled.iloc[-config.holdout_length :].copy()
    return train, test


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Load and prepare data
    df = load_series(config)
    print(f"Loaded {len(df)} data points")
    
    train, test = prepare_data(df, config)
    print(f"Train: {len(train)} points, Test: {len(test)} points")
    
    # Create and fit StatsForecast model
    print("\nFitting StatsForecast AutoARIMA...")
    models = [AutoARIMA(season_length=config.season_length)]
    sf = StatsForecast(models=models, freq=config.freq, n_jobs=1)
    
    sf.fit(train)
    
    # Generate forecast
    print("Generating forecast...")
    forecasts = sf.predict(h=config.prediction_length)
    
    # Evaluate
    aligned_forecast = forecasts["AutoARIMA"].values
    aligned_test = test["y"].values[:len(aligned_forecast)]
    
    if len(aligned_forecast) == len(aligned_test):
        mae_val = mae(aligned_test, aligned_forecast)
        mape_val = mape(aligned_test, aligned_forecast)
        print(f"\nEvaluation Metrics:")
        print(f"  MAE: {mae_val:.4f}")
        print(f"  MAPE: {mape_val:.4f}%")
    
    # Create visualization
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(train["ds"], train["y"], "k-", lw=1.5, label="Historical", alpha=0.8)
    ax.plot(test["ds"], test["y"], "b-", lw=1.5, label="Actual (Test)", alpha=0.8)
    
    forecast_index = pd.date_range(
        start=train["ds"].iloc[-1] + pd.Timedelta(hours=1),
        periods=config.prediction_length,
        freq=config.freq
    )
    ax.plot(forecast_index, aligned_forecast, "r--", lw=2.0, label="StatsForecast AutoARIMA", alpha=0.8)
    
    ax.set_title("StatsForecast AutoARIMA Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, config.output_dir / "statsforecast_forecast.png", dpi=300)
    plt.close(fig)
    print(f" Plot saved -> {config.output_dir / 'statsforecast_forecast.png'}")
    
    print("\n StatsForecast analysis complete")


if __name__ == "__main__":
    main()
