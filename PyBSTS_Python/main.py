#!/usr/bin/env python3
"""
Bayesian Structural Time Series (BSTS) using pybsts
Alternative BSTS implementation using the pybsts library.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pybsts

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.evaluator import Evaluator


def create_bsts_model(data: pd.Series, config: dict):
    """Create and configure BSTS model."""
    specification = {
        "ar_order": config["model"]["ar_order"],
        "local_trend": {"local_level": config["model"]["local_level"]},
        "sigma_prior": np.std(data.values, ddof=1),
        "initial_value": data.values[0],
    }
    
    if config["model"].get("local_slope", False):
        specification["local_trend"]["local_slope"] = True
    
    if config["model"].get("seasonal_period", None):
        specification["seasonal"] = {"nseasons": config["model"]["seasonal_period"]}
    
    model = pybsts.PyBsts(
        config["model"]["distribution"],
        specification,
        {
            "ping": config["model"]["ping"],
            "niter": config["model"]["niter"],
            "burn": config["model"]["burn"],
            "forecast_horizon": config["model"]["forecast_horizon"],
            "seed": config["model"].get("random_seed", 1),
        },
    )
    
    return model


def fit_and_forecast(model, data: pd.Series, config: dict):
    """Fit BSTS model and generate forecasts."""
    model.fit(data.values, seed=config["model"].get("random_seed", 1))
    
    forecast = model.predict(seed=config["model"].get("random_seed", 1))
    forecast_mean = np.mean(forecast, axis=0)
    forecast_std = np.std(forecast, axis=0)
    
    return forecast_mean, forecast_std, forecast


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
    
    # Split train/test using consolidated evaluator
    evaluator = Evaluator(test_size=config.get("evaluation", {}).get("test_size", 0.2))
    train, test = evaluator.split(series)
    print(f"\nTrain: {len(train)} points, Test: {len(test)} points")
    
    # Create and fit BSTS model
    print("\nCreating BSTS model...")
    model = create_bsts_model(train, config)
    
    print("Fitting BSTS model (this may take a while)...")
    forecast_mean, forecast_std, forecast_samples = fit_and_forecast(model, train, config)
    
    # Create forecast index
    forecast_horizon = config["model"]["forecast_horizon"]
    forecast_index = pd.date_range(
        start=train.index[-1] + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=pd.infer_freq(train.index) or "D"
    )
    
    forecast_series = pd.Series(forecast_mean, index=forecast_index)
    forecast_upper = pd.Series(forecast_mean + 2 * forecast_std, index=forecast_index)
    forecast_lower = pd.Series(forecast_mean - 2 * forecast_std, index=forecast_index)
    
    # Evaluate if test data available
    if len(test) >= forecast_horizon:
        test_values = test.iloc[:forecast_horizon].values
        aligned_test = pd.Series(test_values, index=forecast_index[:len(test_values)])
        valid_idx = ~aligned_test.isna() & ~forecast_series.isna()
        
        if valid_idx.sum() > 0:
            mae = mean_absolute_error(aligned_test[valid_idx], forecast_series[valid_idx])
            rmse = np.sqrt(mean_squared_error(aligned_test[valid_idx], forecast_series[valid_idx]))
            print(f"\nEvaluation Metrics:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
    
    # Create visualization
    print("\nCreating visualization...")
    from src import create_forecast_plot
    
    fig, ax = create_forecast_plot(
        train=train[-100:] if len(train) > 100 else train,
        test=test[:forecast_horizon] if len(test) >= forecast_horizon else test,
        forecast=forecast_series,
        conf_int=pd.DataFrame({
            "lower": forecast_lower,
            "upper": forecast_upper
        }),
        figsize=tuple(config.get("plotting", {}).get("figure_size", [12, 6])),
        title="BSTS Forecast",
    )
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / "bsts_forecast.png", dpi=300)
    print(f"Plot saved to: {output_dir / 'bsts_forecast.png'}")
    
    print("\n BSTS analysis complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
