#!/usr/bin/env python3
"""
Moving Average: Simple Forecasting
Simple moving average and exponential moving average forecasting.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    create_forecast_plot,
    save_plot,
    ensure_output_dir,
    get_output_dir,
)
from src.evaluator import Evaluator


def simple_moving_average(data: pd.Series, window: int) -> pd.Series:
    """Calculate simple moving average."""
    return data.rolling(window=window).mean()


def exponential_moving_average(data: pd.Series, alpha: float) -> pd.Series:
    """Calculate exponential moving average."""
    return data.ewm(alpha=alpha, adjust=False).mean()


def weighted_moving_average(data: pd.Series, window: int, weights: list = None) -> pd.Series:
    """Calculate weighted moving average."""
    if weights is None:
        weights = np.ones(window) / window
    return data.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)


def create_forecast(data: pd.Series, config: dict) -> tuple:
    """
    Create moving average forecast.
    
    Returns:
    --------
    tuple
        (moving_average_series, forecast_series)
    """
    method = config["model"]["method"]
    window = config["model"]["window"]
    
    # Calculate moving average
    if method == "SMA":
        ma = simple_moving_average(data, window)
    elif method == "EMA":
        alpha = config["model"].get("alpha", 0.3)
        ma = exponential_moving_average(data, alpha)
    elif method == "WMA":
        weights = config["model"].get("weights")
        ma = weighted_moving_average(data, window, weights)
    else:
        ma = simple_moving_average(data, window)
    
    # Create forecast (constant value = last MA value)
    forecast_horizon = config["model"]["forecast_horizon"]
    last_value = ma.iloc[-1]
    
    # Infer frequency from data
    if len(data.index) > 1:
        freq = pd.infer_freq(data.index) or (data.index[1] - data.index[0])
    else:
        freq = "D"
    
    forecast_index = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=freq
    )
    
    forecast = pd.Series([last_value] * forecast_horizon, index=forecast_index)
    
    return ma, forecast


def main():
    """Main execution function."""
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data using consolidated loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_column", "date"),
        value_column=config["data"].get("value_column", "value")
    )
    
    print(f"Loaded {len(series)} data points")
    print(f"Date range: {series.index.min()} to {series.index.max()}")
    
    # Split into train/test for evaluation
    evaluator = Evaluator(test_size=config.get("evaluation", {}).get("test_size", 0.2))
    train, test = evaluator.split(series)
    print(f"\nTrain: {len(train)} points")
    print(f"Test: {len(test)} points")
    
    # Create forecast
    print(f"\nCalculating {config['model']['method']} forecast...")
    ma, forecast = create_forecast(train, config)
    
    # Align forecast with test period for evaluation
    forecast_aligned = forecast.reindex(test.index, method="nearest")
    valid_idx = ~forecast_aligned.isna() & ~test.isna()
    
    if valid_idx.sum() > 0:
        metrics = evaluator.evaluate(forecast_aligned[valid_idx], test[valid_idx])
        print(f"\nEvaluation Results:")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"Evaluation points: {metrics['n_points']}")
    else:
        metrics = {}
    
    # Create plot using consolidated plotting utility
    print("\nCreating visualization...")
    figsize = config.get("plotting", {}).get("figure_size", [12, 6])
    if isinstance(figsize, dict):
        figsize = figsize.get("figsize", [12, 6])
    
    fig, ax = create_forecast_plot(
        train=train,
        test=test if len(test) > 0 else None,
        forecast=forecast,
        figsize=tuple(figsize),
        title=f"{config['model']['method']} Forecast" + (f" (RMSE: {metrics.get('RMSE', 0):.4f})" if metrics else ""),
    )
    
    # Also plot moving average on historical data
    ax.plot(
        ma.index,
        ma.values,
        "b:",
        linewidth=1.0,
        label=f"{config['model']['method']} (historical)",
        alpha=0.6,
    )
    ax.legend(loc="best")
    
    # Save plot using consolidated utility
    if config.get("output", {}).get("save_plots", True):
        script_dir = Path(__file__).parent
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        plot_path = save_plot(
            fig,
            output_dir / "moving_average_forecast.png",
            dpi=config.get("output", {}).get("dpi", 300)
        )
        print(f"Plot saved to: {plot_path}")
    
    print("\n Moving average forecasting complete")
    
    # Show plot if configured
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
