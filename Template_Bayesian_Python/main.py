#!/usr/bin/env python3
"""
Bayesian Time Series: PyMC
Bayesian time series modeling using PyMC for probabilistic forecasting.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
import pymc as pm
import arviz as az

# Apply SignalPlot's clean defaults
signalplot.apply()


def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj



ts_utils = repo_import("utils.ts_utils")
load_ts_data = ts_utils.load_ts_data
split_ts = ts_utils.split_ts


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_bayesian_model(data, config):
    """Create Bayesian time series model."""
    model_type = config["model"]["type"]

    model_map = {
        "AR1": lambda: create_ar1_model(data, config),
        "RandomWalk": lambda: create_randomwalk_model(data, config),
        "LinearTrend": lambda: create_linear_trend_model(data, config),
    }

    return model_map.get(model_type, model_map["AR1"])()


def create_ar1_model(data, config):
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


def create_randomwalk_model(data, config):
    """Create Random Walk Bayesian model."""
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=1)
        init_dist = pm.Normal.dist(0, 10)
        y_obs = pm.GaussianRandomWalk(
            "y_obs", sigma=sigma, init_dist=init_dist, shape=len(data)
        )
        y_like = pm.Normal("y_like", mu=y_obs, sigma=sigma, observed=data.values)
    return model


def create_linear_trend_model(data, config):
    """Create Linear Trend Bayesian model."""
    x = np.arange(len(data))
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        slope = pm.Normal("slope", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = intercept + slope * x
        y_like = pm.Normal("y_like", mu=mu, sigma=sigma, observed=data.values)
    return model


def sample_posterior(model, config):
    """Sample from posterior distribution."""
    with model:
        trace = pm.sample(
            draws=config["model"].get("draws", 1000),
            tune=config["model"].get("tune", 1000),
            return_inferencedata=True,
            cores=1,
        )
        posterior_predictive = pm.sample_posterior_predictive(trace)
    return trace, posterior_predictive


def create_visualizations(data, trace, posterior_predictive, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=config)

    posterior_samples = posterior_predictive.posterior_predictive["y_like"].values
    [
        posterior_samples.__setitem__(slice(None), posterior_samples[:, 0, :])
        for _ in [None]
        if posterior_samples.ndim > 2
    ]

    posterior_mean = posterior_samples.mean(axis=0)
    lower_bound = np.percentile(posterior_samples, 2.5, axis=0)
    upper_bound = np.percentile(posterior_samples, 97.5, axis=0)

    ax.plot(
        data.index,
        data.values,
        c=config["plotting"]["style"]["colors"]["primary"],
        linewidth=config["plotting"]["style"]["linewidth"],
        alpha=config["plotting"]["style"]["alpha"],
        label="Observed",
    )

    ax.plot(
        data.index,
        posterior_mean,
        c=config["plotting"]["style"]["colors"]["secondary"],
        linewidth=config["plotting"]["style"]["linewidth"],
        label="Posterior Mean",
    )

    ax.fill_between(
        data.index,
        lower_bound,
        upper_bound,
        alpha=0.2,
        color=config["plotting"]["style"]["colors"]["secondary"],
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()

    plt.tight_layout()

    [
        fig.savefig(output_dir / "bayesian_forecast.png", dpi=300, bbox_inches="tight", facecolor="white")
        for _ in [None]
        if config["output"]["save_plots"]
    ]
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    data = load_ts_data(
        Path(__file__).parent.parent / "data" / config["data"]["input_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["value_col"],
    )

    model = create_bayesian_model(data, config)
    trace, posterior_predictive = sample_posterior(model, config)
    create_visualizations(data, trace, posterior_predictive, config)

    print("✓ Bayesian time series modeling complete")


if __name__ == "__main__":
    main()
