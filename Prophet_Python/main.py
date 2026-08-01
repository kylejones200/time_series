#!/usr/bin/env python3
"""
Prophet: Facebook's Time Series Forecasting
Automatic forecasting procedure for business time series.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

from src import BaseTemplate


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
    template = BaseTemplate(config_path="config.yaml", script_dir=Path(__file__).parent)
    config = template.config
    series = template.load_data()

    print(f"Loaded {len(series)} data points")
    print(f"Date range: {series.index.min()} to {series.index.max()}")

    df = prepare_data(series)

    print("\nFitting Prophet model...")
    model = create_prophet_model(config)
    forecast = fit_and_predict(model, df, config)

    forecast_period = forecast[forecast["ds"] > series.index.max()]
    forecast_series = pd.Series(
        forecast_period["yhat"].values,
        index=pd.to_datetime(forecast_period["ds"]),
    )
    conf_int = pd.DataFrame(
        {
            "lower": forecast_period["yhat_lower"].values,
            "upper": forecast_period["yhat_upper"].values,
        },
        index=forecast_series.index,
    )

    print("\nCreating visualization...")
    fig, ax = template.create_plot(
        train=series,
        forecast=forecast_series,
        conf_int=conf_int,
        figsize=tuple(config["plotting"].get("figure_size", [12, 6])),
        title="Prophet Forecast",
    )

    if config["output"].get("save_plots", True):
        plot_path = template.save_plot(fig, "prophet_forecast.png")
        print(f"Plot saved to: {plot_path}")

    print("\n Prophet forecasting complete")

    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
