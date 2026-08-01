#!/usr/bin/env python3
"""Amazon Chronos forecasting template using consolidated utilities."""

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
import torch

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.config import parse_common_config

from chronos import ChronosPipeline
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    resample_rule: Optional[str]
    frequency: Optional[str]
    context_length: Optional[int]
    prediction_length: int
    model_name: str
    device_map: str
    torch_dtype: str
    num_samples: int
    output_dir: Path
    tufte_cfg: dict
    forecast_plot_cfg: dict


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    common = parse_common_config(config_dict, script_dir)

    
    plotting_cfg = config_dict.get("plotting", {})
    tufte_cfg = plotting_cfg.get("tufte", {})
    forecast_cfg = plotting_cfg.get("forecast", {})
    
    return Config(
        data_path=common.data_path,
        date_col=common.date_col,
        value_col=common.value_col,
        resample_rule=config_dict["data"].get("resample_rule"),
        frequency=config_dict["data"].get("frequency"),
        context_length=config_dict["model"].get("context_length"),
        prediction_length=config_dict["model"]["prediction_length"],
        model_name=config_dict["model"]["huggingface_model"],
        device_map=config_dict["model"].get("device_map", "cpu"),
        torch_dtype=config_dict["model"].get("torch_dtype", "float32"),
        num_samples=config_dict["model"].get("num_samples", 20),
        output_dir=common.output_dir,
        tufte_cfg=tufte_cfg,
        forecast_plot_cfg=forecast_cfg,
    )


def load_series(config: Config) -> pd.Series:
    """Load time series using consolidated loader."""
    # Use consolidated loader
    series = load_time_series(
        str(config.data_path),
        date_column=config.date_col,
        value_column=config.value_col
    )
    
    # Apply resampling if configured
    if config.resample_rule:
        series = series.resample(config.resample_rule).mean()
    
    # Apply frequency conversion if needed
    if config.frequency:
        series = series.asfreq(config.frequency)
    
    if len(series) <= config.prediction_length:
        raise ValueError("Time series length must exceed prediction length.")
    
    return series


def prepare_tensors(
    series: pd.Series, config: Config
) -> tuple[torch.Tensor, torch.Tensor, pd.Index]:
    """Prepare tensors for Chronos."""
    pred_len = config.prediction_length
    context_len = config.context_length or len(series) - pred_len
    context_len = min(context_len, len(series) - pred_len)
    
    train_values = series.iloc[-(pred_len + context_len) : -pred_len].values
    test_values = series.iloc[-pred_len:].values
    test_index = series.index[-pred_len:]
    
    dtype_attr = getattr(torch, config.torch_dtype, None)
    if dtype_attr is None:
        raise ValueError(f"Unsupported torch dtype: {config.torch_dtype}")
    
    context_tensor = torch.tensor(train_values, dtype=dtype_attr)
    if context_tensor.ndim == 1:
        context_tensor = context_tensor.unsqueeze(0)
    
    return context_tensor, torch.tensor(test_values, dtype=dtype_attr), test_index


def load_pipeline(config: Config) -> ChronosPipeline:
    """Load Chronos pipeline."""
    kwargs = {
        "device_map": config.device_map,
        "torch_dtype": getattr(torch, config.torch_dtype, torch.float32),
    }
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        kwargs["use_auth_token"] = hf_token
    
    pipeline = ChronosPipeline.from_pretrained(config.model_name, **kwargs)
    pipeline.eval()
    return pipeline


def generate_forecast(
    pipeline: ChronosPipeline, context: torch.Tensor, config: Config
) -> np.ndarray:
    """Generate forecast from Chronos pipeline."""
    with torch.no_grad():
        forecast = pipeline.predict(
            context, config.prediction_length, num_samples=config.num_samples
        )
    return forecast.cpu().numpy()


def compute_stats(forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute forecast statistics."""
    low, median, high = np.quantile(forecast[0], [0.1, 0.5, 0.9], axis=0)
    return low, median, high


def compute_metrics(true: torch.Tensor, median: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    true_np = true.cpu().numpy()
    mae = mean_absolute_error(true_np, median)
    rmse = np.sqrt(mean_squared_error(true_np, median))
    mape = np.mean(np.abs((true_np - median) / true_np)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def plot_forecast(
    series: pd.Series,
    forecast_index: pd.Index,
    median: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    metrics: dict,
    config: Config,
) -> None:
    """Plot Chronos forecast."""
    history_window = 7 * config.prediction_length
    history = series.iloc[-history_window:]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(
        history.index, history.values, color="royalblue", linewidth=2, label="History"
    )
    ax.plot(
        forecast_index, median, color="tomato", linewidth=2, label="Chronos Forecast"
    )
    ax.fill_between(
        forecast_index, low, high, color="tomato", alpha=0.3, label="80% interval"
    )
    
    plot_title = config.forecast_plot_cfg.get("title", "Chronos Forecast")
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, config.output_dir / "chronos_forecast.png", dpi=300)
    plt.close(fig)
    print(f" Chronos plot saved -> {config.output_dir / 'chronos_forecast.png'}")


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
    
    # Prepare tensors
    context, test_tensor, test_index = prepare_tensors(series, config)
    
    # Load pipeline
    print(f"\nLoading Chronos model: {config.model_name}")
    pipeline = load_pipeline(config)
    
    # Generate forecast
    print("Generating forecast...")
    forecast = generate_forecast(pipeline, context, config)
    low, median, high = compute_stats(forecast)
    
    # Compute metrics
    metrics = compute_metrics(test_tensor, median)
    print(f"\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Plot forecast
    print("\nCreating visualization...")
    plot_forecast(series, test_index, median, low, high, metrics, config)
    
    print("\n Chronos forecasting complete")


if __name__ == "__main__":
    main()
