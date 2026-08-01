#!/usr/bin/env python3
"""
Box-Jenkins Methodology for ARIMA Modeling
Systematic approach to ARIMA model identification, estimation, and diagnostics.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.evaluator import Evaluator


def find_differencing_order(timeseries, max_d=3):
    """Find optimal differencing order using ADF test."""
    current_series = timeseries.copy()
    for d in range(max_d + 1):
        result = adfuller(current_series.dropna(), autolag="AIC")
        if result[1] <= 0.05:
            return d, current_series
        current_series = current_series.diff()
    return max_d, current_series


def fit_arima_model(train_data, config):
    """Fit ARIMA model using auto_arima or manual parameters."""
    if config["model"]["use_auto_arima"]:
        model = auto_arima(
            train_data,
            start_p=config["model"]["start_p"],
            start_q=config["model"]["start_q"],
            max_p=config["model"]["max_p"],
            max_q=config["model"]["max_q"],
            d=config["model"]["d"],
            seasonal=config["model"]["seasonal"],
            stepwise=True,
            suppress_warnings=True,
            trace=False,
            error_action="ignore",
        )
    else:
        order = (config["model"]["p"], config["model"]["d"], config["model"]["q"])
        model = ARIMA(train_data, order=order).fit()
    return model


def main():
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data using consolidated loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value")
    )
    
    # Split train/test using consolidated evaluator
    evaluator = Evaluator(test_size=config.get("data", {}).get("test_size", 0.2))
    train, test = evaluator.split(series)
    train_data = train
    
    d_order, differenced_data = find_differencing_order(
        train_data, max_d=config["model"]["max_d"]
    )
    print(f"\nOptimal differencing order (d): {d_order}")
    
    if config["model"]["d"] is None:
        config["model"]["d"] = d_order
    
    model = fit_arima_model(train_data, config)
    print(f"\nBest Model: ARIMA{model.order}")
    print(f"AIC: {model.aic():.2f}")
    
    residuals = model.resid()
    lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    print("\nLjung-Box Test for Residual Autocorrelation:")
    print(lb_test)
    
    forecast_result = model.predict(
        n_periods=len(test), return_conf_int=True, alpha=0.05
    )
    forecast = forecast_result[0]
    conf_int = forecast_result[1]
    
    # Create forecast Series with confidence intervals
    forecast_series = pd.Series(forecast, index=test.index)
    conf_int_df = pd.DataFrame(conf_int, index=test.index, columns=["lower", "upper"])
    
    mae = mean_absolute_error(test.values, forecast)
    rmse = np.sqrt(mean_squared_error(test.values, forecast))
    r2 = r2_score(test.values, forecast)
    
    print(f"\nModel Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Diagnostics plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(train_data.index, train_data.values, "k-", linewidth=1.5)
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)
    axes[0, 0].set_title("Original Series")
    
    axes[0, 1].plot(
        differenced_data.index, differenced_data.values, "r-", linewidth=1.5
    )
    axes[0, 1].axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Differenced")
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)
    axes[0, 1].set_title("Differenced Series")
    
    plot_acf(differenced_data.dropna(), lags=40, ax=axes[1, 0])
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)
    axes[1, 0].set_title("ACF")
    
    plot_pacf(differenced_data.dropna(), lags=40, ax=axes[1, 1])
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)
    axes[1, 1].set_title("PACF")
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    dpi = config.get("plotting", {}).get("dpi", 300)
    save_plot(fig, output_dir / "box_jenkins_diagnostics.png", dpi=dpi)
    
    # Forecast plot using consolidated plotting utility
    from src import create_forecast_plot
    
    fig, ax = create_forecast_plot(
        train=train_data[-100:] if len(train_data) > 100 else train_data,
        test=test,
        forecast=forecast_series,
        conf_int=conf_int_df,
        figsize=tuple(config.get("plotting", {}).get("figure_size", [12, 6])),
        title=f"Box-Jenkins Forecast (ARIMA{model.order}) - RMSE: {rmse:.4f}",
    )
    
    plt.tight_layout()
    save_plot(fig, output_dir / "box_jenkins_forecast.png", dpi=300)
    
    print("\n Box-Jenkins analysis complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
