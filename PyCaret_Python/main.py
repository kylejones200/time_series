#!/usr/bin/env python3
"""
PyCaret: Low-Code Time Series Forecasting
Automated time series forecasting with minimal code.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from pycaret.time_series import *


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data using consolidated loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value")
    )
    
    print(f"Loaded {len(series)} data points")
    
    # Setup PyCaret time series environment
    print("\nSetting up PyCaret time series environment...")
    s = setup(
        data=series,
        fh=config["model"]["forecast_horizon"],
        session_id=config["model"].get("session_id", 42),
        verbose=False,
    )
    
    # Compare models
    print("Comparing models...")
    best_model = compare_models(
        include=config["model"].get("models", ["arima", "exp_smooth", "theta"]),
        sort=config["model"].get("sort_metric", "MAE"),
        verbose=False,
    )
    
    print(f"\nBest model: {best_model}")
    
    # Finalize model
    print("Finalizing model...")
    final_model = finalize_model(best_model)
    
    # Generate forecast
    print("Generating forecast...")
    forecast = predict_model(final_model, fh=config["model"]["forecast_horizon"])
    
    # Create visualization
    print("\nCreating visualization...")
    plot_model(final_model, plot="forecast", save=True, verbose=False)
    
    # Also create custom plot
    fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))
    
    ax.plot(
        series.index[-100:] if len(series) > 100 else series.index,
        series.values[-100:] if len(series) > 100 else series.values,
        "k-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label="Historical",
    )
    
    # Create forecast index
    forecast_horizon = config["model"]["forecast_horizon"]
    forecast_index = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=pd.infer_freq(series.index) or "D"
    )
    
    forecast_values = forecast.values.flatten()[:forecast_horizon]
    ax.plot(
        forecast_index,
        forecast_values,
        "r--",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        label="PyCaret Forecast",
    )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("PyCaret Time Series Forecast")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if config.get("output", {}).get("save_plots", True):
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "pycaret_forecast.png", dpi=300)
        print(f"Plot saved to: {output_dir / 'pycaret_forecast.png'}")
    
    print("\n PyCaret forecasting complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
