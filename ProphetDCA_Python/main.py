#!/usr/bin/env python3
"""Prophet + Decline Curve Analysis: Direct comparison of time series forecasting vs DCA."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False
        warnings.warn("Prophet not available. Install with: pip install prophet")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import consolidated utilities
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

# Import DCA models
from models.dca import ArpsHyperbolic, ArpsExponential, ArpsHarmonic

warnings.filterwarnings("ignore")


def fit_prophet(
    series: pd.Series,
    forecast_horizon: int,
    seasonality_mode: str = "multiplicative",
) -> tuple:
    """
    Fit Prophet model and generate forecast.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with datetime index
    forecast_horizon : int
        Number of periods to forecast
    seasonality_mode : str
        'multiplicative' or 'additive' (default: 'multiplicative')
    
    Returns:
    --------
    tuple
        (prophet_model, forecast_df, forecast_series)
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is required for this template")
    
    # Prepare data for Prophet
    df = pd.DataFrame({
        "ds": series.index,
        "y": series.values,
    })
    
    # Fit Prophet
    model = Prophet(seasonality_mode=seasonality_mode)
    model.fit(df)
    
    # Create future dataframe
    freq = pd.infer_freq(series.index) or "D"
    future = model.make_future_dataframe(periods=forecast_horizon, freq=freq)
    
    # Forecast
    forecast_df = model.predict(future)
    
    # Extract forecast series
    forecast_series = pd.Series(
        forecast_df["yhat"].values,
        index=pd.to_datetime(forecast_df["ds"]),
    )
    
    return model, forecast_df, forecast_series


def fit_dca(
    series: pd.Series,
    dca_type: str = "hyperbolic",
) -> tuple:
    """
    Fit decline curve analysis model.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with datetime index
    dca_type : str
        DCA type: "hyperbolic", "exponential", or "harmonic"
    
    Returns:
    --------
    tuple
        (dca_model, fitted_series, forecast_series)
    """
    # Convert to days since start
    days_since_start = (series.index - series.index[0]).days.values
    values = series.values
    
    # Fit DCA model
    if dca_type == "hyperbolic":
        model = ArpsHyperbolic()
    elif dca_type == "exponential":
        model = ArpsExponential()
    elif dca_type == "harmonic":
        model = ArpsHarmonic()
    else:
        raise ValueError(f"Unknown DCA type: {dca_type}")
    
    # Fit on historical data
    model.fit(days_since_start, values)
    
    # Generate fitted values
    fitted_values = model.predict(days_since_start)
    fitted_series = pd.Series(fitted_values, index=series.index)
    
    # Generate forecast (extend to future)
    forecast_horizon = len(series)  # Forecast same length as historical
    future_days = np.arange(
        days_since_start[-1] + 1,
        days_since_start[-1] + 1 + forecast_horizon,
    )
    forecast_values = model.predict(future_days)
    
    # Create forecast index
    last_date = series.index[-1]
    freq = pd.infer_freq(series.index) or "D"
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=freq,
    )
    forecast_series = pd.Series(forecast_values, index=forecast_dates)
    
    return model, fitted_series, forecast_series


def main():
    """Main execution function."""
    if not PROPHET_AVAILABLE:
        print("ERROR: Prophet is not installed.")
        print("Install with: pip install prophet")
        sys.exit(1)
    
    script_dir = Path(__file__).parent
    config = load_config(script_dir / "config.yaml")
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    
    # Load data
    data_config = config["data"]
    repo_root = script_dir.parent
    data_path = repo_root / data_config["input_file"]
    series = load_time_series(
        str(data_path),
        date_column=data_config.get("date_column", "date"),
        value_column=data_config.get("value_column", "value"),
    )
    
    # Split data
    test_size = config["evaluation"].get("test_size", 0.2)
    split_idx = int(len(series) * (1 - test_size))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:] if split_idx < len(series) else None
    
    print(f"Loaded {len(series)} data points")
    print(f"Train: {len(train)}, Test: {len(test) if test is not None else 0}")
    
    # Fit Prophet
    forecast_horizon = config["evaluation"].get("forecast_horizon", len(test) if test is not None else 24)
    print("\nFitting Prophet model...")
    prophet_model, prophet_forecast_df, prophet_forecast = fit_prophet(
        train,
        forecast_horizon=forecast_horizon,
        seasonality_mode=config.get("prophet", {}).get("seasonality_mode", "multiplicative"),
    )
    
    # Fit DCA
    dca_type = config.get("dca", {}).get("type", "hyperbolic")
    print(f"\nFitting {dca_type} decline curve...")
    dca_model, dca_fitted, dca_forecast = fit_dca(train, dca_type=dca_type)
    
    # Evaluate on test set if available
    prophet_metrics = {}
    dca_metrics = {}
    
    if test is not None and len(test) > 0:
        # Align forecasts with test
        prophet_test = prophet_forecast[prophet_forecast.index.isin(test.index)]
        dca_test = dca_forecast[dca_forecast.index.isin(test.index)]
        test_aligned = test[test.index.isin(prophet_test.index)]
        
        if len(prophet_test) > 0:
            prophet_mse = mean_squared_error(test_aligned, prophet_test)
            prophet_mae = mean_absolute_error(test_aligned, prophet_test)
            prophet_rmse = np.sqrt(prophet_mse)
            prophet_r2 = r2_score(test_aligned, prophet_test)
            
            prophet_metrics = {
                "RMSE": prophet_rmse,
                "MAE": prophet_mae,
                "R²": prophet_r2,
                "MSE": prophet_mse,
            }
        
        if len(dca_test) > 0:
            dca_mse = mean_squared_error(test_aligned, dca_test)
            dca_mae = mean_absolute_error(test_aligned, dca_test)
            dca_rmse = np.sqrt(dca_mse)
            dca_r2 = r2_score(test_aligned, dca_test)
            
            dca_metrics = {
                "RMSE": dca_rmse,
                "MAE": dca_mae,
                "R²": dca_r2,
                "MSE": dca_mse,
            }
        
        print(f"\nProphet Performance:")
        for metric, value in prophet_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\n{dca_type.capitalize()} DCA Performance:")
        for metric, value in dca_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full comparison
    axes[0].plot(train.index, train.values, "k-", linewidth=2, label="Historical (Train)", alpha=0.8)
    if test is not None:
        axes[0].plot(test.index, test.values, "g-", linewidth=2, label="Actual (Test)", alpha=0.8)
    
    # Prophet forecast
    axes[0].plot(
        prophet_forecast.index,
        prophet_forecast.values,
        "b--",
        linewidth=2,
        label="Prophet Forecast",
        alpha=0.8,
    )
    
    # Prophet confidence intervals
    if "yhat_lower" in prophet_forecast_df.columns:
        prophet_lower = pd.Series(
            prophet_forecast_df["yhat_lower"].values,
            index=pd.to_datetime(prophet_forecast_df["ds"]),
        )
        prophet_upper = pd.Series(
            prophet_forecast_df["yhat_upper"].values,
            index=pd.to_datetime(prophet_forecast_df["ds"]),
        )
        axes[0].fill_between(
            prophet_forecast.index,
            prophet_lower[prophet_forecast.index].values,
            prophet_upper[prophet_forecast.index].values,
            color="blue",
            alpha=0.2,
            label="Prophet 95% CI",
        )
    
    # DCA forecast
    axes[0].plot(
        dca_forecast.index,
        dca_forecast.values,
        "r--",
        linewidth=2,
        label=f"{dca_type.capitalize()} DCA Forecast",
        alpha=0.8,
    )
    
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Prophet vs Decline Curve Analysis Forecast")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: DCA fitted vs actual
    axes[1].plot(train.index, train.values, "k-", linewidth=2, label="Actual", alpha=0.8)
    axes[1].plot(dca_fitted.index, dca_fitted.values, "r--", linewidth=2, label=f"{dca_type.capitalize()} DCA Fit", alpha=0.8)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Value")
    axes[1].set_title(f"{dca_type.capitalize()} Decline Curve Fit")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / config["output"].get("plot_file", "prophet_dca_comparison.png")
    save_plot(fig, plot_path, dpi=config["output"].get("dpi", 300))
    print(f"\nPlot saved to: {plot_path}")
    
    # Save forecasts
    forecast_df = pd.DataFrame({
        "date": prophet_forecast.index,
        "prophet_forecast": prophet_forecast.values,
        "dca_forecast": dca_forecast[dca_forecast.index.isin(prophet_forecast.index)].values,
    })
    
    csv_path = output_dir / config["output"].get("forecast_file", "prophet_dca_forecast.csv")
    forecast_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Forecast saved to: {csv_path}")
    
    # Save metrics
    if prophet_metrics and dca_metrics:
        metrics_df = pd.DataFrame({
            "model": ["Prophet", f"{dca_type.capitalize()} DCA"],
            "RMSE": [prophet_metrics["RMSE"], dca_metrics["RMSE"]],
            "MAE": [prophet_metrics["MAE"], dca_metrics["MAE"]],
            "R²": [prophet_metrics["R²"], dca_metrics["R²"]],
        })
        
        metrics_path = output_dir / config["output"].get("metrics_file", "prophet_dca_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
        print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

