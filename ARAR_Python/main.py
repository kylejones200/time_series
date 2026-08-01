#!/usr/bin/env python3
"""
ARAR: Autoregressive Autoregressive
Refactored to support config-driven workflows, evaluation, and comparison with ARIMA.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.evaluator import Evaluator

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf


def resolve_data_path(input_file: str) -> Path:
    """Resolve data file path relative to repo root."""
    path = Path(input_file)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "data" / path
    if candidate.exists():
        return candidate
    return path


def load_series(config: Dict) -> pd.Series:
    """Load time series with optional resampling."""
    # Use consolidated loader
    series = load_time_series(
        str(resolve_data_path(config["data"]["input_file"])),
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value"),
    ).sort_index()

    # Apply resampling if configured
    resample_cfg = config["data"].get("resample", {})
    if resample_cfg.get("enabled"):
        from utils.ts_utils import resample_ts
        series = resample_ts(
            series,
            freq=resample_cfg.get("freq", "D"),
            method=resample_cfg.get("method", "mean"),
        )

    return series


def fit_arar_model(train_data: pd.Series, config: Dict) -> AutoReg:
    """Fit ARAR model."""
    max_lag = config["model"].get("max_lag", 5)
    model = AutoReg(train_data, lags=max_lag).fit()
    return model


def fit_arima_model(train_data: pd.Series, config: Dict):
    """Fit ARIMA model for comparison."""
    order = tuple(config["model"].get("arima_order", (1, 1, 1)))
    model = ARIMA(train_data, order=order).fit()
    return model


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data
    series = load_series(config)
    print(f"Loaded {len(series)} data points")
    
    # Split train/test using consolidated evaluator
    evaluator = Evaluator(test_size=config.get("evaluation", {}).get("test_size", 0.2))
    train, test = evaluator.split(series)
    
    # Fit models
    print("\nFitting ARAR model...")
    arar_model = fit_arar_model(train, config)
    
    print("\nFitting ARIMA model for comparison...")
    arima_model = fit_arima_model(train, config)
    
    # Generate forecasts
    arar_forecast = arar_model.forecast(steps=len(test))
    arima_forecast = arima_model.forecast(steps=len(test))
    
    # Align forecasts with test index
    arar_forecast_series = pd.Series(arar_forecast, index=test.index)
    arima_forecast_series = pd.Series(arima_forecast, index=test.index)
    
    # Evaluate
    arar_mae = mean_absolute_error(test, arar_forecast_series)
    arar_rmse = np.sqrt(mean_squared_error(test, arar_forecast_series))
    arar_mape = mean_absolute_percentage_error(test, arar_forecast_series)
    
    arima_mae = mean_absolute_error(test, arima_forecast_series)
    arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast_series))
    arima_mape = mean_absolute_percentage_error(test, arima_forecast_series)
    
    print(f"\nARAR - MAE: {arar_mae:.4f}, RMSE: {arar_rmse:.4f}, MAPE: {arar_mape:.4f}%")
    print(f"ARIMA - MAE: {arima_mae:.4f}, RMSE: {arima_rmse:.4f}, MAPE: {arima_mape:.4f}%")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(train.index, train.values, "k-", linewidth=1.5, label="Historical (Train)", alpha=0.8)
    ax.plot(test.index, test.values, "g-", linewidth=1.5, label="Actual (Test)", alpha=0.8)
    ax.plot(test.index, arar_forecast_series.values, "r--", linewidth=1.5, label=f"ARAR Forecast (MAE: {arar_mae:.4f})", alpha=0.8)
    ax.plot(test.index, arima_forecast_series.values, "b--", linewidth=1.5, label=f"ARIMA Forecast (MAE: {arima_mae:.4f})", alpha=0.8)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("ARAR vs ARIMA Forecast Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    # Save plot using consolidated utility
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / "arar_comparison.png", dpi=300)
    print(f"\nPlot saved to: {output_dir / 'arar_comparison.png'}")
    
    print("\n ARAR forecasting complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
