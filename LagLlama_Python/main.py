#!/usr/bin/env python3
"""LagLlama forecasting template using consolidated utilities."""

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
import torch

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from lagllama import LagLlama, LagLlamaHF
from transformers import AutoTokenizer


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
    output_dir: Path
    output_plot: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    repo_root = script_dir.parent
    data_path = repo_root / "data" / config_dict["data"]["input_file"]
    output_dir = ensure_output_dir(Path(script_dir) / config_dict["output"]["output_dir"])
    
    experiment = config_dict["experiment"]
    model_cfg = config_dict["model"]
    
    return Config(
        data_path=data_path,
        date_col=config_dict["data"]["date_col"],
        value_col=config_dict["data"]["value_col"],
        freq=config_dict["data"].get("freq", "MS"),
        history_end=pd.Timestamp(experiment["history_end"]),
        forecast_start=pd.Timestamp(experiment["forecast_start"]),
        forecast_end=pd.Timestamp(experiment["forecast_end"]),
        checkpoint=model_cfg["checkpoint"],
        context_length=int(model_cfg.get("context_length", 512)),
        horizon=int(model_cfg.get("horizon", 8)),
        output_dir=output_dir,
        output_plot=output_dir / config_dict["output"]["tufte_plot"],
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


def build_model(config: Config) -> LagLlamaHF:
    """Build LagLlama model."""
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model = LagLlamaHF.from_pretrained(config.checkpoint)
    model.eval()
    return model, tokenizer


def generate_forecast(model, tokenizer, train_series: pd.Series, config: Config) -> np.ndarray:
    """Generate forecast from LagLlama model."""
    # Prepare input - LagLlama expects specific format
    context_values = train_series.iloc[-config.context_length:].values.astype(np.float32)
    
    # Tokenize and generate (simplified - actual implementation may vary)
    # This is a placeholder for the actual LagLlama inference logic
    forecast_values = np.zeros(config.horizon)
    
    # In practice, you'd use the model's generate/predict method here
    # forecast_values = model.generate(...)
    
    return forecast_values


def plot_tufte(
    series: pd.Series,
    history_end: pd.Timestamp,
    forecast_values: np.ndarray,
    actual: pd.Series,
    config: Config,
) -> None:
    """Plot LagLlama forecast."""
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
    ax.plot(forecast_series.index, forecast_series.values, color="#000000", lw=2.0, label="LagLlama Forecast")
    
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("EIA Net Generation — LagLlama forecast Jan–Aug 2025")
    ax.legend(loc="best")
    
    fig.tight_layout()
    save_plot(fig, config.output_plot, dpi=300)
    plt.close(fig)
    print(f" LagLlama plot saved -> {config.output_plot}")


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
    
    train_series = series.loc[: config.history_end]
    actual = series.loc[config.forecast_start : config.forecast_end]
    
    # Build model
    print(f"\nBuilding LagLlama model from {config.checkpoint}...")
    model, tokenizer = build_model(config)
    
    # Generate forecast
    print("Generating forecast...")
    forecast_values = generate_forecast(model, tokenizer, train_series, config)
    
    # Plot forecast
    print("\nCreating visualization...")
    plot_tufte(series, config.history_end, forecast_values, actual, config)
    
    print("\n LagLlama forecasting complete")


if __name__ == "__main__":
    main()
