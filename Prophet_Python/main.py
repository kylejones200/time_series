#!/usr/bin/env python3
"""
Prophet: Facebook's Time Series Forecasting
Automatic forecasting procedure for business time series.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    create_forecast_plot,
    save_plot,
    ensure_output_dir,
    get_output_dir,
)


def prepare_data(data: pd.Series) -> pd.DataFrame:
    """Prepare data for Prophet (requires 'ds' and 'y' columns)."""
    return pd.DataFrame({"ds": data.index, "y": data.values})


def create_prophet_model(config: dict) -> Prophet:
    """Create Prophet model with config parameters."""
    model_params = {
        "yearly_seasonality": config["model"].get("yearly_seasonality", "auto"),
        "weekly_seasonality": config["model"].get("weekly_seasonality", "auto"),
        "daily_seasonality": config["model"].get("daily_seasonality", False),
        "seasonality_mode": config["model"].get("seasonality_mode", "additive"),
        "growth": config["model"].get("growth", "linear"),
    }
    
    # Merge with any additional params
    model_params.update(config["model"].get("params", {}))
    
    return Prophet(**model_params)


def fit_and_predict(model: Prophet, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fit model and generate predictions."""
    model.fit(df)
    
    forecast_horizon = config["model"]["forecast_horizon"]
    future = model.make_future_dataframe(periods=forecast_horizon)
    forecast = model.predict(future)
    
    return forecast


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
    
    # Prepare data for Prophet
    df = prepare_data(series)
    
    # Create and fit model
    print("\nFitting Prophet model...")
    model = create_prophet_model(config)
    forecast = fit_and_predict(model, df, config)
    
    # Extract forecast components
    forecast_period = forecast[forecast["ds"] > series.index.max()]
    historical_period = forecast[forecast["ds"] <= series.index.max()]
    
    # Create forecast Series with confidence intervals
    forecast_series = pd.Series(
        forecast_period["yhat"].values,
        index=pd.to_datetime(forecast_period["ds"])
    )
    
    conf_int = pd.DataFrame({
        "lower": forecast_period["yhat_lower"].values,
        "upper": forecast_period["yhat_upper"].values,
    }, index=forecast_series.index)
    
    # Create plot using consolidated plotting utility
    print("\nCreating visualization...")
    fig, ax = create_forecast_plot(
        train=series,
        forecast=forecast_series,
        conf_int=conf_int,
        figsize=tuple(config["plotting"].get("figure_size", [12, 6])),
        title="Prophet Forecast",
    )
    
    # Save plot using consolidated utility
    if config["output"].get("save_plots", True):
        script_dir = Path(__file__).parent
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        plot_path = save_plot(
            fig,
            output_dir / "prophet_forecast.png",
            dpi=config.get("output", {}).get("dpi", 300)
        )
        print(f"Plot saved to: {plot_path}")
    
    print("\n Prophet forecasting complete")
    
    # Show plot if configured
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
