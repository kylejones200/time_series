#!/usr/bin/env python3
"""
Bayesian Time Series: PyMC
Bayesian time series modeling using PyMC for probabilistic forecasting.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.evaluator import Evaluator


def create_bayesian_model(data: pd.Series, config: dict):
    """Create Bayesian time series model."""
    model_type = config["model"]["type"]
    
    model_map = {
        "AR1": lambda: create_ar1_model(data, config),
        "RandomWalk": lambda: create_randomwalk_model(data, config),
        "LinearTrend": lambda: create_linear_trend_model(data, config),
    }
    
    return model_map.get(model_type, model_map["AR1"])()


def create_ar1_model(data: pd.Series, config: dict):
    """Create AR(1) Bayesian model."""
    with pm.Model() as model:
        phi = pm.Normal("phi", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        init_dist = pm.Normal.dist(0, 10)
        y_obs = pm.GaussianRandomWalk(
            "y_obs", sigma=sigma, init_dist=init_dist, shape=len(data)
        )
        y_like = pm.Normal("y_like", mu=y_obs, sigma=sigma, observed=data.values)
    return model


def create_randomwalk_model(data: pd.Series, config: dict):
    """Create Random Walk Bayesian model."""
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=1)
        init_dist = pm.Normal.dist(0, 10)
        y_obs = pm.GaussianRandomWalk(
            "y_obs", sigma=sigma, init_dist=init_dist, shape=len(data)
        )
        y_like = pm.Normal("y_like", mu=y_obs, sigma=sigma, observed=data.values)
    return model


def create_linear_trend_model(data: pd.Series, config: dict):
    """Create Linear Trend Bayesian model."""
    x = np.arange(len(data))
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        slope = pm.Normal("slope", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = intercept + slope * x
        y_like = pm.Normal("y_like", mu=mu, sigma=sigma, observed=data.values)
    return model


def sample_posterior(model, config: dict):
    """Sample from posterior distribution."""
    with model:
        trace = pm.sample(
            draws=config["model"].get("draws", 1000),
            tune=config["model"].get("tune", 1000),
            return_inferencedata=True,
            random_seed=config.get("random_seed", 42),
        )
    return trace


def generate_forecast(trace, model, n_periods: int):
    """Generate forecast from posterior samples."""
    with model:
        forecast = pm.sample_posterior_predictive(trace, predictions=True, var_names=["y_like"])
    return forecast


def create_visualizations(data: pd.Series, trace, config: dict, script_dir: Path):
    """Generate visualizations."""
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    
    # Posterior plots
    fig = az.plot_trace(trace, compact=True)
    plt.tight_layout()
    save_plot(fig[0][0].figure, output_dir / "bayesian_trace.png", dpi=300)
    plt.close()
    
    # Forecast plot
    fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))
    ax.plot(data.index, data.values, "k-", linewidth=1.5, label="Historical", alpha=0.8)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"Bayesian {config['model']['type']} Model")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, output_dir / "bayesian_forecast.png", dpi=300)
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


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
    print(f"Date range: {series.index.min()} to {series.index.max()}")
    
    # Split train/test if needed
    if config.get("evaluation", {}).get("test_size", 0) > 0:
        evaluator = Evaluator(test_size=config["evaluation"]["test_size"])
        train, test = evaluator.split(series)
        series = train  # Use train data for modeling
        print(f"Using {len(train)} points for training, {len(test)} for testing")
    
    # Create and fit Bayesian model
    print(f"\nCreating Bayesian {config['model']['type']} model...")
    model = create_bayesian_model(series, config)
    
    print("Sampling from posterior...")
    trace = sample_posterior(model, config)
    
    print(" Posterior sampling complete")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(series, trace, config, script_dir)
    
    print("\n Bayesian time series analysis complete")


if __name__ == "__main__":
    main()
