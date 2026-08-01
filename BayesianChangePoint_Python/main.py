#!/usr/bin/env python3
"""
Bayesian Change Point Detection
Bayesian MCMC approach to detect change points in time series using PyMC.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

from scipy import stats


def detect_change_point_bayesian(data: pd.Series, config: dict):
    """Detect change point using Bayesian MCMC."""
    n = len(data)
    
    with pm.Model() as model:
        tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)
        
        lambda_1 = pm.Exponential("lambda_1", lam=config["model"]["lambda_prior"])
        lambda_2 = pm.Exponential("lambda_2", lam=config["model"]["lambda_prior"])
        
        idx = np.arange(n)
        lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)
        
        observation = pm.Poisson("obs", mu=lambda_, observed=data.values)
        
        trace = pm.sample(
            config["model"]["draws"],
            tune=config["model"]["tune"],
            chains=config["model"]["chains"],
            cores=config["model"]["cores"],
            return_inferencedata=True,
            random_seed=config["model"]["random_seed"],
        )
    
    return trace, model


def detect_change_point_frequentist(data: pd.Series, config: dict):
    """Detect change point using frequentist method (for comparison)."""
    n = len(data)
    best_tau = 0
    best_p_value = 1.0
    
    for tau in range(1, n - 1):
        before = data.iloc[:tau]
        after = data.iloc[tau:]
        
        if len(before) > 1 and len(after) > 1:
            stat, p_value = stats.ks_2samp(before, after)
            if p_value < best_p_value:
                best_p_value = p_value
                best_tau = tau
    
    return best_tau, best_p_value


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
    
    # Bayesian change point detection
    if config["model"].get("method", "bayesian") == "bayesian":
        print("\nDetecting change point using Bayesian MCMC...")
        trace, model = detect_change_point_bayesian(series, config)
        
        # Extract posterior samples of tau
        tau_samples = trace.posterior["tau"].values.flatten()
        tau_mean = int(np.mean(tau_samples))
        tau_std = int(np.std(tau_samples))
        
        print(f"\nBayesian Change Point Detection:")
        print(f"  Mean change point index: {tau_mean}")
        print(f"  Std deviation: {tau_std}")
        print(f"  Estimated change point date: {series.index[tau_mean]}")
        
        # Create visualization
        print("\nCreating visualization...")
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        axes[0].plot(series.index, series.values, "k-", lw=1.5, alpha=0.8, label="Time Series")
        axes[0].axvline(series.index[tau_mean], color="r", linestyle="--", lw=2, label=f"Change Point (index: {tau_mean})")
        axes[0].fill_between(series.index, series.index[tau_mean - tau_std], series.index[tau_mean + tau_std], alpha=0.2, color="r", label="Uncertainty")
        axes[0].set_ylabel("Value")
        axes[0].set_title("Bayesian Change Point Detection")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)
        
        # Plot posterior distribution of tau
        axes[1].hist(tau_samples, bins=50, alpha=0.7, color="b", edgecolor="k")
        axes[1].axvline(tau_mean, color="r", linestyle="--", lw=2, label=f"Mean: {tau_mean}")
        axes[1].set_xlabel("Change Point Index")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Posterior Distribution of Change Point")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "bayesian_change_point.png", dpi=300)
        print(f"Plot saved to: {output_dir / 'bayesian_change_point.png'}")
        plt.close(fig)
    
    # Frequentist change point detection (for comparison)
    if config["model"].get("frequentist_comparison", False):
        print("\nDetecting change point using frequentist method...")
        best_tau, best_p_value = detect_change_point_frequentist(series, config)
        
        print(f"\nFrequentist Change Point Detection:")
        print(f"  Change point index: {best_tau}")
        print(f"  p-value: {best_p_value:.4f}")
        print(f"  Estimated change point date: {series.index[best_tau]}")
    
    print("\n Bayesian change point detection complete")


if __name__ == "__main__":
    main()
