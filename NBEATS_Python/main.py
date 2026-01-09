#!/usr/bin/env python3
"""N-BEATS evaluation using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from typing import Tuple

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

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
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
    input_chunk_length: int
    output_chunk_length: int
    n_epochs: int
    output_dir: Path
    output_plot: Path


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
        horizon=int(config_dict["model"]["horizon"]),
        n_splits=int(config_dict["model"]["n_splits"]),
        input_chunk_length=int(config_dict["model"]["input_chunk_length"]),
        output_chunk_length=int(config_dict["model"]["output_chunk_length"]),
        n_epochs=int(config_dict["model"]["n_epochs"]),
        output_dir=output_dir,
        output_plot=output_dir / config_dict["output"]["tufte_plot"],
    )


def load_series(config: Config) -> TimeSeries:
    """Load time series into Darts TimeSeries format."""
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")
    
    df = pd.read_csv(config.data_path, header=0)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not present in CSV")
    
    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.value_col])
    df = df.sort_values(config.date_col)
    series = pd.Series(
        pd.to_numeric(df[config.value_col], errors="coerce").dropna().values,
        index=pd.DatetimeIndex(df[config.date_col]),
    )
    series = series.asfreq(config.freq).astype(float)
    return TimeSeries.from_series(series)


def build_model(config: Config) -> NBEATSModel:
    """Build N-BEATS model."""
    return NBEATSModel(
        input_chunk_length=config.input_chunk_length,
        output_chunk_length=config.output_chunk_length,
        n_epochs=config.n_epochs,
        random_state=42,
        pl_trainer_kwargs={
            "enable_progress_bar": False,
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
        },
    )


def rolling_origin_nbeats(
    series: TimeSeries, config: Config
) -> Tuple[float, TimeSeries, TimeSeries]:
    """Rolling origin evaluation for N-BEATS."""
    values = series.to_series()
    idx = np.arange(len(values))
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    maes = []
    last_true = None
    last_pred = None
    
    scaler = Scaler()
    series_scaled = scaler.fit_transform(series)
    
    for train_idx, _ in splitter.split(idx):
        end_idx = train_idx[-1]
        train_series_scaled = series_scaled[: end_idx + 1]
        future_series = series[end_idx + 1 : end_idx + 1 + config.horizon]
        
        if len(future_series) < config.horizon:
            continue
        
        model = build_model(config)
        model.fit(train_series_scaled)
        
        forecast_scaled = model.predict(config.horizon)
        forecast = scaler.inverse_transform(forecast_scaled)
        
        forecast_series = forecast.to_series()
        actual_series = future_series.to_series()
        
        mae = mean_absolute_error(actual_series.values, forecast_series.values)
        maes.append(mae)
        
        last_true = future_series
        last_pred = forecast
    
    mean_mae = float(np.mean(maes)) if maes else float("nan")
    print(f"N-BEATS rolling-origin MAE: {mean_mae:.3f}")
    return mean_mae, last_true, last_pred


def plot_nbeats_forecast(series: TimeSeries, config: Config, last_forecast: TimeSeries) -> None:
    """Plot N-BEATS forecast."""
    history_end = pd.Timestamp("2024-12-01")
    forecast_start = pd.Timestamp("2025-01-01")
    
    history = series[:history_end]
    actual = series[forecast_start:]
    
    scaler = Scaler()
    series_scaled = scaler.fit_transform(series[:history_end])
    
    model = build_model(config)
    model.fit(series_scaled)
    
    forecast_scaled = model.predict(config.horizon)
    forecast = scaler.inverse_transform(forecast_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.to_series().index, history.to_series().values, color="#555555", lw=1.5, label="History")
    ax.axvline(forecast_start, color="#777777", linestyle="--", lw=1)
    
    if not actual.is_empty:
        ax.plot(actual.to_series().index, actual.to_series().values, color="#1f77b4", lw=1.8, label="Actual")
    
    ax.plot(forecast.to_series().index, forecast.to_series().values, color="red", lw=2.0, label="N-BEATS Forecast")
    
    ax.set_title("N-BEATS Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    save_plot(fig, config.output_plot, dpi=300)
    plt.close(fig)
    print(f" N-BEATS plot saved -> {config.output_plot}")


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
    _, last_true, last_pred = rolling_origin_nbeats(series, config)
    
    # Plot forecast
    if last_pred is not None:
        plot_nbeats_forecast(series, config, last_pred)
    
    print("\n N-BEATS analysis complete")


if __name__ == "__main__":
    main()
