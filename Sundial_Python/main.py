#!/usr/bin/env python3
"""Moirai forecasting template using consolidated utilities."""

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
import torch

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.config import parse_common_config

from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    history_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    checkpoint: str
    context_length: int
    horizon: int
    num_samples: int
    output_dir: Path
    output_plot: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    common = parse_common_config(config_dict, script_dir)

    
    experiment = config_dict["experiment"]
    model_cfg = config_dict["model"]
    
    return Config(
        data_path=common.data_path,
        date_col=common.date_col,
        value_col=common.value_col,
        freq=config_dict["data"].get("freq", "MS"),
        history_end=pd.Timestamp(experiment["history_end"]),
        forecast_start=pd.Timestamp(experiment["forecast_start"]),
        forecast_end=pd.Timestamp(experiment["forecast_end"]),
        checkpoint=model_cfg["checkpoint"],
        context_length=int(model_cfg.get("context_length", 512)),
        horizon=int(model_cfg.get("horizon", 8)),
        num_samples=int(model_cfg.get("num_samples", 100)),
        output_dir=common.output_dir,
        output_plot=common.output_dir / config_dict["output"]["tufte_plot"],
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


def build_moirai(config: Config) -> MoiraiForecast:
    """Build Moirai model."""
    module = MoiraiModule.from_pretrained(config.checkpoint)
    forecast_model = MoiraiForecast(
        prediction_length=config.horizon,
        target_dim=1,
        context_length=config.context_length,
        module=module,
        num_samples=config.num_samples,
    )
    return forecast_model


def generate_forecast(
    model: MoiraiForecast, train_series: pd.Series, config: Config
) -> np.ndarray:
    """Generate forecast from Moirai model."""
    dataset = ListDataset(
        [
            {
                "target": train_series.values.astype(np.float32),
                "start": train_series.index[0],
            }
        ],
        freq=config.freq,
    )
    predictor = model.create_predictor(batch_size=1, device="cpu")
    prediction_iterator = predictor.predict(dataset)
    
    forecast_array = None
    for forecast in prediction_iterator:
        if hasattr(forecast, "mean"):
            forecast_array = forecast.mean
        else:
            forecast_array = forecast.samples.mean(axis=0)
        break
    
    if forecast_array is None:
        raise RuntimeError("Moirai did not return any forecasts.")
    
    return np.asarray(forecast_array).reshape(-1)


def plot_tufte(
    series: pd.Series,
    history_end: pd.Timestamp,
    forecast_values: np.ndarray,
    actual: pd.Series,
    config: Config,
) -> None:
    """Plot Moirai forecast."""
    start_2024 = pd.Timestamp("2024-01-01")
    history = series.loc[start_2024:history_end]
    forecast_index = pd.period_range(
        config.forecast_start, config.forecast_end, freq="M"
    ).to_timestamp()
    forecast_series = pd.Series(
        forecast_values[: len(forecast_index)], index=forecast_index
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, color="#888888", lw=1.5, label="History")
    ax.axvline(config.forecast_start, color="#666666", linestyle="--", lw=1)
    if not actual.empty:
        ax.plot(actual.index, actual.values, color="#444444", lw=1.8, label="Actual")
    ax.plot(forecast_series.index, forecast_series.values, color="#000000", lw=2.0, label="Moirai Forecast")
    
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("EIA Net Generation — Moirai forecast Jan–Aug 2025")
    ax.legend(loc="best")
    
    fig.tight_layout()
    save_plot(fig, config.output_plot, dpi=300)
    plt.close(fig)
    print(f" Moirai plot saved -> {config.output_plot}")


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Load series
    series = load_series(config)
    print(f"Loaded {len(series)} data points")
    
    train_series = series.loc[: config.history_end]
    actual = series.loc[config.forecast_start : config.forecast_end]
    
    # Build model
    print(f"\nBuilding Moirai model from {config.checkpoint}...")
    model = build_moirai(config)
    
    # Generate forecast
    print("Generating forecast...")
    forecast_values = generate_forecast(model, train_series, config)
    
    # Plot forecast
    print("\nCreating visualization...")
    plot_tufte(series, config.history_end, forecast_values, actual, config)
    
    print("\n Moirai forecasting complete")


if __name__ == "__main__":
    main()
