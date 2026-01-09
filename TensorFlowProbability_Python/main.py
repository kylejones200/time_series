#!/usr/bin/env python3
"""TensorFlow Probability Structural Time Series for probabilistic forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import TensorFlow Probability
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow_probability import sts
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    warnings.warn(
        "TensorFlow Probability not available. Install with: pip install tensorflow tensorflow-probability"
    )

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

warnings.filterwarnings("ignore")


def build_structural_model(
    observed_time_series: np.ndarray,
    num_seasons: int = 12,
    include_trend: bool = True,
    include_seasonal: bool = True,
    include_autoregressive: bool = False,
    ar_order: int = 1,
) -> sts.Sum:
    """
    Build a structural time series model.
    
    Parameters:
    -----------
    observed_time_series : np.ndarray
        Observed time series values
    num_seasons : int
        Number of seasonal periods (e.g., 12 for monthly, 4 for quarterly)
    include_trend : bool
        Whether to include local linear trend component
    include_seasonal : bool
        Whether to include seasonal component
    include_autoregressive : bool
        Whether to include autoregressive component
    ar_order : int
        Order of autoregressive component
    
    Returns:
    --------
    sts.Sum
        Structural time series model
    """
    if not TFP_AVAILABLE:
        raise ImportError("TensorFlow Probability is required for this template")
    
    components = []
    
    if include_trend:
        trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
        components.append(trend)
    
    if include_seasonal:
        seasonal = sts.Seasonal(
            num_seasons=num_seasons,
            observed_time_series=observed_time_series,
        )
        components.append(seasonal)
    
    if include_autoregressive:
        autoregressive = sts.Autoregressive(
            order=ar_order,
            observed_time_series=observed_time_series,
            name="autoregressive",
        )
        components.append(autoregressive)
    
    if not components:
        raise ValueError("At least one component (trend, seasonal, or autoregressive) must be included")
    
    model = sts.Sum(components, observed_time_series=observed_time_series)
    return model


def fit_model(
    model: sts.Sum,
    observed_time_series: np.ndarray,
    num_variational_steps: int = 200,
    learning_rate: float = 0.1,
    num_samples: int = 50,
) -> tuple:
    """
    Fit structural time series model using variational inference.
    
    Parameters:
    -----------
    model : sts.Sum
        Structural time series model
    observed_time_series : np.ndarray
        Observed time series values
    num_variational_steps : int
        Number of optimization steps
    learning_rate : float
        Learning rate for optimizer
    num_samples : int
        Number of posterior samples to draw
    
    Returns:
    --------
    tuple
        (variational_posteriors, parameter_samples, elbo_loss_curve)
    """
    if not TFP_AVAILABLE:
        raise ImportError("TensorFlow Probability is required for this template")
    
    # Build variational surrogate posteriors
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)
    
    # Optimize variational loss
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    @tf.function
    def train():
        elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=model.joint_log_prob(observed_time_series=observed_time_series),
            surrogate_posterior=variational_posteriors,
            optimizer=optimizer,
            num_steps=num_variational_steps,
        )
        return elbo_loss_curve
    
    elbo_loss_curve = train()
    
    # Draw samples from variational posterior
    parameter_samples = variational_posteriors.sample(num_samples)
    
    return variational_posteriors, parameter_samples, elbo_loss_curve


def forecast(
    model: sts.Sum,
    observed_time_series: np.ndarray,
    parameter_samples: dict,
    forecast_horizon: int,
    num_samples: int = 20,
) -> tuple:
    """
    Generate probabilistic forecast.
    
    Parameters:
    -----------
    model : sts.Sum
        Fitted structural time series model
    observed_time_series : np.ndarray
        Observed time series values
    parameter_samples : dict
        Parameter samples from variational posterior
    forecast_horizon : int
        Number of steps to forecast
    num_samples : int
        Number of forecast scenarios to sample
    
    Returns:
    --------
    tuple
        (forecast_mean, forecast_std, forecast_samples)
    """
    if not TFP_AVAILABLE:
        raise ImportError("TensorFlow Probability is required for this template")
    
    # Generate forecast distribution
    forecast_dist = tfp.sts.forecast(
        model=model,
        observed_time_series=observed_time_series,
        parameter_samples=parameter_samples,
        num_steps_forecast=forecast_horizon,
    )
    
    # Extract mean, stddev, and samples
    forecast_mean = forecast_dist.mean().numpy()[..., 0]
    forecast_std = forecast_dist.stddev().numpy()[..., 0]
    forecast_samples = forecast_dist.sample(num_samples).numpy()[..., 0]
    
    return forecast_mean, forecast_std, forecast_samples


def main():
    """Main execution function."""
    if not TFP_AVAILABLE:
        print("ERROR: TensorFlow Probability is not installed.")
        print("Install with: pip install tensorflow tensorflow-probability")
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
    evaluator = Evaluator(test_size=config["evaluation"].get("test_size", 0.2))
    train, test = evaluator.split(series)
    
    # Convert to numpy for TFP
    train_values = train.values.astype(np.float32)
    test_values = test.values.astype(np.float32)
    
    # Build model
    model_config = config.get("model", {})
    print("Building structural time series model...")
    model = build_structural_model(
        observed_time_series=train_values,
        num_seasons=model_config.get("num_seasons", 12),
        include_trend=model_config.get("include_trend", True),
        include_seasonal=model_config.get("include_seasonal", True),
        include_autoregressive=model_config.get("include_autoregressive", False),
        ar_order=model_config.get("ar_order", 1),
    )
    print(f"Model components: {[c.name for c in model.components]}")
    
    # Fit model
    print("Fitting model with variational inference...")
    variational_posteriors, parameter_samples, elbo_loss = fit_model(
        model=model,
        observed_time_series=train_values,
        num_variational_steps=model_config.get("num_variational_steps", 200),
        learning_rate=model_config.get("learning_rate", 0.1),
        num_samples=model_config.get("num_samples", 50),
    )
    print(f"ELBO loss (final): {elbo_loss[-1]:.2f}")
    
    # Generate forecast
    forecast_horizon = config["evaluation"].get("forecast_horizon", len(test))
    print(f"Generating {forecast_horizon}-step forecast...")
    forecast_mean, forecast_std, forecast_samples = forecast(
        model=model,
        observed_time_series=train_values,
        parameter_samples=parameter_samples,
        forecast_horizon=forecast_horizon,
        num_samples=model_config.get("forecast_samples", 20),
    )
    
    # Create forecast index
    last_date = train.index[-1]
    freq = pd.infer_freq(train.index) or "D"
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=freq,
    )
    
    forecast_series = pd.Series(forecast_mean, index=forecast_dates)
    forecast_std_series = pd.Series(forecast_std, index=forecast_dates)
    
    # Create confidence intervals (95% CI = mean ± 1.96 * std)
    conf_int = pd.DataFrame({
        "lower": forecast_mean - 1.96 * forecast_std,
        "upper": forecast_mean + 1.96 * forecast_std,
    }, index=forecast_dates)
    
    # Evaluate on test set (if forecast horizon matches test length)
    if len(forecast_series) == len(test):
        mse = mean_squared_error(test.values, forecast_mean)
        mae = mean_absolute_error(test.values, forecast_mean)
        rmse = np.sqrt(mse)
        r2 = r2_score(test.values, forecast_mean)
        
        print(f"\nTest Set Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Mean Uncertainty (σ): {forecast_std.mean():.4f}")
    
    # Create plot
    fig, ax = create_forecast_plot(
        train=train,
        test=test if len(test) <= len(forecast_series) else None,
        forecast=forecast_series,
        conf_int=conf_int,
        title="TensorFlow Probability Structural Time Series Forecast",
        xlabel="Date",
        ylabel="Value",
        train_label="Historical (Train)",
        test_label="Actual (Test)",
        forecast_label="TFP Forecast",
        show_ci=True,
    )
    
    # Save plot
    plot_path = output_dir / config["output"].get("plot_file", "tfp_forecast.png")
    save_plot(fig, plot_path, dpi=config["output"].get("dpi", 300))
    print(f"\nPlot saved to: {plot_path}")
    
    # Save forecast to CSV
    forecast_df = pd.DataFrame({
        "date": forecast_series.index,
        "forecast": forecast_mean,
        "std": forecast_std,
        "lower_95": conf_int["lower"].values,
        "upper_95": conf_int["upper"].values,
    })
    
    csv_path = output_dir / config["output"].get("forecast_file", "tfp_forecast.csv")
    forecast_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Forecast saved to: {csv_path}")
    
    # Save metrics if available
    if len(forecast_series) == len(test):
        metrics_df = pd.DataFrame({
            "metric": ["RMSE", "MAE", "R²", "Mean_Uncertainty"],
            "value": [rmse, mae, r2, forecast_std.mean()],
        })
        
        metrics_path = output_dir / config["output"].get("metrics_file", "tfp_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
        print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

