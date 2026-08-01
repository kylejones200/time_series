#!/usr/bin/env python3
"""Gaussian Process Regression for time series forecasting with uncertainty quantification."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import consolidated utilities
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    create_forecast_plot,
    save_plot,
)
from src.evaluator import Evaluator
from src.run_logger import append_run_log, utc_now_iso

warnings.filterwarnings("ignore")


def build_time_features(
    index: pd.DatetimeIndex, reference_start: pd.Timestamp
) -> np.ndarray:
    """Build trend + seasonal time features for GP."""
    days = (index - reference_start).days.values.astype(float)
    day_of_year = index.dayofyear.values.astype(float)
    month = index.month.values.astype(float)
    # Cyclical seasonal features
    doy_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    mon_sin = np.sin(2 * np.pi * month / 12.0)
    mon_cos = np.cos(2 * np.pi * month / 12.0)
    return np.column_stack([days, doy_sin, doy_cos, mon_sin, mon_cos])


def prepare_time_series_features(
    series: pd.Series, reference_start: pd.Timestamp | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert time series to feature matrix for GP.
    
    For time series, we use the time index as the single feature dimension.
    This is converted to numeric (days since start) for the GP model.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with datetime index
    
    Returns:
    --------
    tuple
        (X, y) where X is (n_samples, 1) time features, y is values
    """
    # Convert datetime index to numeric using a shared origin date.
    # Using different origins for train vs. test breaks temporal continuity.
    if reference_start is None:
        reference_start = series.index[0]
    time_numeric = build_time_features(series.index, reference_start)
    values = series.values
    
    return time_numeric, values


def fit_gaussian_process(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel_type: str = "rbf_matern",
    alpha: float = 0.1,
    n_restarts: int = 3,
    random_state: int = 42,
) -> tuple[GaussianProcessRegressor, StandardScaler, StandardScaler]:
    """
    Fit Gaussian Process Regression model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features (time as numeric)
    y_train : np.ndarray
        Training target values
    kernel_type : str
        Kernel type: "rbf", "matern", or "rbf_matern" (default)
    alpha : float
        Noise level (regularization)
    n_restarts : int
        Number of optimizer restarts
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        (gpr_model, x_scaler, y_scaler)
    """
    # Scale features and targets
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # Define kernel
    if kernel_type == "rbf":
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e1))
    elif kernel_type == "matern":
        kernel = Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-3, 1e1))
    else:  # rbf_matern (default)
        # Combined kernel: RBF for smoothness + Matern for flexibility
        kernel = (
            RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e1)) +
            Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-3, 1e1))
        )
    
    # Add white noise kernel
    kernel += WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-5, 1e1))
    
    # Fit GP model
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,  # Noise handled by WhiteKernel
        n_restarts_optimizer=n_restarts,
        random_state=random_state,
    )
    
    gpr.fit(X_train_scaled, y_train_scaled)
    
    return gpr, x_scaler, y_scaler


def forecast_gp(
    gpr: GaussianProcessRegressor,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    train_index: pd.DatetimeIndex,
    forecast_horizon: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate forecast with uncertainty bounds.
    
    Parameters:
    -----------
    gpr : GaussianProcessRegressor
        Fitted GP model
    x_scaler : StandardScaler
        Feature scaler
    y_scaler : StandardScaler
        Target scaler
    train_index : pd.DatetimeIndex
        Training data index (for frequency inference)
    forecast_horizon : int
        Number of steps to forecast
    
    Returns:
    --------
    tuple
        (forecast_mean, forecast_std) as pd.Series with datetime index
    """
    # Create future time points
    last_date = train_index[-1]
    freq = pd.infer_freq(train_index) or "D"
    
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=freq,
    )
    
    # Build future features using same shared origin date.
    first_date = train_index[0]
    future_numeric = build_time_features(future_dates, first_date)
    
    # Scale and predict
    future_scaled = x_scaler.transform(future_numeric)
    pred_scaled, sigma_scaled = gpr.predict(future_scaled, return_std=True)
    
    # Transform back to original scale
    pred_orig = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    # Note: sigma is in scaled space, approximate conversion
    # For exact conversion, we'd need the full covariance, but this approximation works
    sigma_orig = sigma_scaled * y_scaler.scale_[0]
    
    # Create Series with datetime index
    forecast_mean = pd.Series(pred_orig, index=future_dates)
    forecast_std = pd.Series(sigma_orig, index=future_dates)
    
    return forecast_mean, forecast_std


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    config = load_config(script_dir / "config.yaml")
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    started_at = utc_now_iso()
    t0 = time.perf_counter()

    metrics_log: dict[str, float] = {}
    details_log: dict[str, str] = {}
    status = "success"
    error_msg = None

    try:
        # Load data
        data_config = config["data"]
        # Resolve data path relative to repo root
        repo_root = script_dir.parent
        data_path = repo_root / data_config["input_file"]
        series = load_time_series(
            str(data_path),
            date_column=data_config.get("date_column", "date"),
            value_column=data_config.get("value_column", "value"),
        )
        details_log["data_path"] = str(data_path)

        # Split data
        evaluator = Evaluator(test_size=config["evaluation"].get("test_size", 0.2))
        train, test = evaluator.split(series)

        # Prepare features
        ref_start = train.index[0]
        X_train, y_train = prepare_time_series_features(train, reference_start=ref_start)
        X_test, y_test = prepare_time_series_features(test, reference_start=ref_start)

        # Fit GP model
        model_config = config.get("model", {})
        print("Training Gaussian Process Regression model...")
        gpr, x_scaler, y_scaler = fit_gaussian_process(
            X_train,
            y_train,
            kernel_type=model_config.get("kernel_type", "rbf_matern"),
            alpha=model_config.get("alpha", 0.1),
            n_restarts=model_config.get("n_restarts", 3),
            random_state=model_config.get("random_state", 42),
        )

        print(f"Trained kernel: {gpr.kernel_}")
        details_log["trained_kernel"] = str(gpr.kernel_)

        # Evaluate on test set
        X_test_scaled = x_scaler.transform(X_test)
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        y_pred_scaled, sigma_test_scaled = gpr.predict(X_test_scaled, return_std=True)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        sigma_test = sigma_test_scaled * y_scaler.scale_[0]

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics_log = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mean_uncertainty_sigma": float(sigma_test.mean()),
        }

        print(f"\nTest Set Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Mean Uncertainty (σ): {sigma_test.mean():.4f}")

        # Generate forecast
        forecast_horizon = config["evaluation"].get("forecast_horizon", len(test))
        forecast_mean, forecast_std = forecast_gp(
            gpr, x_scaler, y_scaler, train.index, forecast_horizon
        )

        # Create confidence intervals (95% CI = mean ± 1.96 * std)
        conf_int = pd.DataFrame(
            {
                "lower": forecast_mean - 1.96 * forecast_std,
                "upper": forecast_mean + 1.96 * forecast_std,
            },
            index=forecast_mean.index,
        )

        # Create plot
        fig, ax = create_forecast_plot(
            train=train,
            test=test,
            forecast=forecast_mean,
            conf_int=conf_int,
            title="Gaussian Process Regression Forecast",
            xlabel="Date",
            ylabel="Value",
            train_label="Historical (Train)",
            test_label="Actual (Test)",
            forecast_label="GP Forecast",
            show_ci=True,
        )

        # Save plot
        plot_path = output_dir / config["output"].get("plot_file", "gp_forecast.png")
        save_plot(fig, plot_path, dpi=config["output"].get("dpi", 300))
        print(f"\nPlot saved to: {plot_path}")

        # Save forecast to CSV
        forecast_df = pd.DataFrame(
            {
                "date": forecast_mean.index,
                "forecast": forecast_mean.values,
                "std": forecast_std.values,
                "lower_95": conf_int["lower"].values,
                "upper_95": conf_int["upper"].values,
            }
        )

        csv_path = output_dir / config["output"].get("forecast_file", "gp_forecast.csv")
        forecast_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Forecast saved to: {csv_path}")

        # Save metrics
        metrics_df = pd.DataFrame(
            {
                "metric": ["RMSE", "MAE", "R²", "Mean_Uncertainty"],
                "value": [rmse, mae, r2, sigma_test.mean()],
            }
        )

        metrics_path = output_dir / config["output"].get("metrics_file", "gp_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
        print(f"Metrics saved to: {metrics_path}")

    except Exception as e:
        status = "failed"
        error_msg = str(e)
        raise
    finally:
        ended_at = utc_now_iso()
        duration = time.perf_counter() - t0
        log_path = append_run_log(
            output_dir=output_dir,
            script_name="GaussianProcess_Python",
            started_at_utc=started_at,
            ended_at_utc=ended_at,
            duration_seconds=duration,
            status=status,
            metrics=metrics_log,
            details=details_log,
            error=error_msg,
        )
        print(f"Run log saved to: {log_path}")


if __name__ == "__main__":
    main()

