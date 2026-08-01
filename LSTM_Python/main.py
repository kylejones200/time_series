#!/usr/bin/env python3
"""LSTM evaluation using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
from src.config import parse_common_config

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
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
    epochs: int
    output_dir: Path
    output_plot: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    common = parse_common_config(config_dict, script_dir)

    
    return Config(
        data_path=common.data_path,
        date_col=common.date_col,
        value_col=common.value_col,
        freq=config_dict["data"].get("freq", "MS"),
        horizon=int(config_dict["model"]["horizon"]),
        n_splits=int(config_dict["model"]["n_splits"]),
        input_chunk_length=int(config_dict["model"]["input_chunk_length"]),
        output_chunk_length=int(config_dict["model"]["output_chunk_length"]),
        epochs=int(config_dict["model"]["epochs"]),
        output_dir=common.output_dir,
        output_plot=common.output_dir / config_dict["output"]["tufte_plot"],
    )


def load_series(config: Config) -> TimeSeries:
    """Load time series into Darts TimeSeries format."""
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")
    
    df = pd.read_csv(config.data_path, encoding="utf-8")
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


def build_model(config: Config) -> RNNModel:
    """Build LSTM model."""
    return RNNModel(
        model="LSTM",
        input_chunk_length=config.input_chunk_length,
        output_chunk_length=config.output_chunk_length,
        training_length=max(config.input_chunk_length, 24),
        n_rnn_layers=2,
        hidden_dim=64,
        n_epochs=config.epochs,
        random_state=42,
        pl_trainer_kwargs={
            "enable_progress_bar": False,
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
        },
    )


def rolling_origin_lstm(
    series: TimeSeries, config: Config
) -> Tuple[float, TimeSeries, TimeSeries]:
    """Rolling origin evaluation for LSTM."""
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
    print(f"LSTM rolling-origin MAE: {mean_mae:.3f}")
    return mean_mae, last_true, last_pred


def plot_lstm_forecast(series: TimeSeries, config: Config, last_forecast: TimeSeries) -> None:
    """Plot LSTM forecast."""
    history_end = pd.Timestamp("2024-12-01")
    forecast_start = pd.Timestamp("2025-01-01")
    forecast_index = pd.period_range(
        forecast_start, periods=config.horizon, freq="M"
    ).to_timestamp()
    
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
    
    if len(actual) > 0:
        ax.plot(actual.to_series().index, actual.to_series().values, color="#1f77b4", lw=1.8, label="Actual")
    
    ax.plot(forecast.to_series().index, forecast.to_series().values, color="red", lw=2.0, label="LSTM Forecast")
    
    ax.set_title("LSTM Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    save_plot(fig, config.output_plot, dpi=300)
    plt.close(fig)
    print(f" LSTM plot saved -> {config.output_plot}")


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
    _, last_true, last_pred = rolling_origin_lstm(series, config)
    
    # Plot forecast
    if last_pred is not None:
        plot_lstm_forecast(series, config, last_pred)
    
    print("\n LSTM analysis complete")


if __name__ == "__main__":
    main()
