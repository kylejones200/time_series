#!/usr/bin/env python3
"""
Volatility Models (ARCH/GARCH)
Volatility forecasting using ARCH and GARCH models for financial time series.
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
    ensure_output_dir,
    get_output_dir,
    save_plot,
    create_forecast_plot,
)
from src.evaluator import Evaluator

from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_volatility_model(data: pd.Series, config: dict):
    """Create ARCH/GARCH volatility model."""
    model_type = config["model"]["type"]
    
    model_map = {
        "ARCH": lambda: arch_model(
            data,
            vol=model_type,
            p=config["model"]["p"],
            q=0,
            dist=config["model"]["distribution"],
        ),
        "GARCH": lambda: arch_model(
            data,
            vol=model_type,
            p=config["model"]["p"],
            q=config["model"]["q"],
            dist=config["model"]["distribution"],
        ),
        "EGARCH": lambda: arch_model(
            data,
            vol="EGARCH",
            p=config["model"]["p"],
            q=config["model"]["q"],
            dist=config["model"]["distribution"],
        ),
    }
    
    return model_map.get(model_type, model_map["GARCH"])()


def fit_and_forecast(model, config: dict):
    """Fit model and generate volatility forecasts."""
    fitted_model = model.fit(
        update_freq=config["model"].get("update_freq", 1),
        disp=config["model"].get("disp", "off"),
    )
    
    forecast = fitted_model.forecast(horizon=config["model"]["forecast_horizon"])
    forecast_variance = forecast.variance.iloc[-1].values
    forecast_volatility = np.sqrt(forecast_variance)
    
    return fitted_model, forecast_variance, forecast_volatility


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
    
    # Calculate returns if configured (typical for volatility modeling)
    if config.get("data", {}).get("use_returns", True):
        returns = series.pct_change().dropna()
        data = returns
        print(f"Using returns: {len(returns)} data points")
    else:
        data = series
    
    # Split train/test using consolidated evaluator
    evaluator = Evaluator(test_size=config.get("evaluation", {}).get("test_size", 0.2))
    train, test = evaluator.split(data)
    print(f"\nTrain: {len(train)} points, Test: {len(test)} points")
    
    # Create and fit volatility model
    model_type = config["model"]["type"]
    print(f"\nFitting {model_type} volatility model...")
    model = create_volatility_model(train, config)
    
    fitted_model, forecast_variance, forecast_volatility = fit_and_forecast(model, config)
    
    print(f"\n{model_type} Model Summary:")
    print(fitted_model.summary())
    
    # Create forecast index
    forecast_horizon = config["model"]["forecast_horizon"]
    forecast_index = pd.date_range(
        start=train.index[-1] + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=pd.infer_freq(train.index) or "D"
    )
    
    forecast_series = pd.Series(forecast_volatility, index=forecast_index)
    
    # Evaluate if we have test data
    if len(test) >= forecast_horizon:
        test_volatility = test.iloc[:forecast_horizon].abs()  # Use absolute returns as proxy for volatility
        aligned_test = test_volatility.reindex(forecast_index, method="nearest")
        valid_idx = ~aligned_test.isna() & ~forecast_series.isna()
        
        if valid_idx.sum() > 0:
            mae = mean_absolute_error(aligned_test[valid_idx], forecast_series[valid_idx])
            rmse = np.sqrt(mean_squared_error(aligned_test[valid_idx], forecast_series[valid_idx]))
            print(f"\nEvaluation Metrics:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
    
    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot returns/values
    axes[0].plot(
        train.index[-100:] if len(train) > 100 else train.index,
        train.values[-100:] if len(train) > 100 else train.values,
        "k-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label="Historical",
    )
    if len(test) > 0:
        axes[0].plot(
            test.index[:forecast_horizon] if len(test) >= forecast_horizon else test.index,
            test.values[:forecast_horizon] if len(test) >= forecast_horizon else test.values,
            "g-",
            linewidth=config.get("plotting", {}).get("linewidth", 1.5),
            alpha=config.get("plotting", {}).get("alpha", 0.8),
            label="Actual (Test)",
        )
    axes[0].set_title(f"{model_type} Model - Returns/Values")
    axes[0].set_ylabel("Return" if config.get("data", {}).get("use_returns", True) else "Value")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    
    # Plot volatility forecast
    axes[1].plot(
        forecast_index,
        forecast_volatility,
        "r-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        label=f"{model_type} Volatility Forecast",
    )
    axes[1].fill_between(
        forecast_index,
        forecast_volatility * 0.8,  # Lower bound approximation
        forecast_volatility * 1.2,  # Upper bound approximation
        alpha=0.2,
        color="r",
        label="Uncertainty",
    )
    axes[1].set_title(f"{model_type} Volatility Forecast")
    axes[1].set_ylabel("Volatility")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / f"{model_type.lower()}_volatility.png", dpi=300)
    print(f"Plot saved to: {output_dir / f'{model_type.lower()}_volatility.png'}")
    
    print(f"\n {model_type} volatility analysis complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
