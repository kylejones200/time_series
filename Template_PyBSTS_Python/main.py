#!/usr/bin/env python3
"""
Bayesian Structural Time Series (BSTS) using pybsts
Alternative BSTS implementation using the pybsts library.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
import pybsts


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
ensure_datetime_index = ts_utils.ensure_datetime_index
split_ts = ts_utils.split_ts


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_bsts_model(data, config):
    """Create and configure BSTS model."""
    specification = {
        "ar_order": config["model"]["ar_order"],
        "local_trend": {"local_level": config["model"]["local_level"]},
        "sigma_prior": np.std(data, ddof=1),
        "initial_value": data[0],
    }

    if config["model"].get("local_slope", False):
        specification["local_trend"]["local_slope"] = True

    if config["model"].get("seasonal_period", None):
        specification["seasonal"] = {"nseasons": config["model"]["seasonal_period"]}

    model = pybsts.PyBsts(
        config["model"]["distribution"],
        specification,
        {
            "ping": config["model"]["ping"],
            "niter": config["model"]["niter"],
            "burn": config["model"]["burn"],
            "forecast_horizon": config["model"]["forecast_horizon"],
            "seed": config["model"].get("random_seed", 1),
        },
    )

    return model


def fit_and_forecast(model, data, config):
    """Fit BSTS model and generate forecasts."""
    model.fit(data, seed=config["model"].get("random_seed", 1))

    forecast = model.predict(seed=config["model"].get("random_seed", 1))
    forecast_mean = np.mean(forecast, axis=0)
    forecast_std = np.std(forecast, axis=0)

    return forecast_mean, forecast_std, forecast


def create_visualizations(
    data, train_data, test_data, forecast_mean, forecast_std, config
):
    """Generate visualizations for BSTS forecast."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=tuple(config["plotting"]["figure_size"]))
    
    ax.plot(
        train_data.index,
        train_data.values,
        "k-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label="Train",
    )

    if test_data is not None:
        ax.plot(
            test_data.index,
            test_data.values,
            "g-",
            linewidth=config["plotting"]["linewidth"],
            alpha=config["plotting"]["alpha"],
            label="Test",
        )

    forecast_index = pd.date_range(
        start=train_data.index[-1] + pd.Timedelta(hours=1),
        periods=len(forecast_mean),
        freq="h",
    )

    ax.plot(
        forecast_index,
        forecast_mean,
        "r--",
        linewidth=config["plotting"]["linewidth"],
        label="Forecast",
    )

    ax.fill_between(
        forecast_index,
        forecast_mean - 1.96 * forecast_std,
        forecast_mean + 1.96 * forecast_std,
        color="red",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    if test_data is not None:
        ax.axvline(
            x=train_data.index[-1],
            color="k",
            linestyle=":",
            linewidth=config["plotting"]["linewidth"],
            label="Train/Test Split",
        )

    ax.set_title(config["plot_titles"]["forecast"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    output_path = output_dir / "pybsts_forecast.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def main():
    """Main execution function."""
    config = load_config()

    df = load_ts_data(
        data_path=Path(__file__).parent.parent / "data" / config["data"]["input_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["value_col"],
    )
    df = ensure_datetime_index(df, time_col=config["data"]["date_col"])

    train_df, test_df = split_ts(df, test_size=config["data"]["test_size"])

    model = create_bsts_model(train_df[config["data"]["value_col"]].values, config)

    print("\nFitting BSTS model...")
    print(f"Iterations: {config['model']['niter']}")
    print(f"Burn-in: {config['model']['burn']}")

    forecast_mean, forecast_std, forecast_samples = fit_and_forecast(
        model, train_df[config["data"]["value_col"]].values, config
    )

    print("\nBSTS Forecast Results:")
    print("=" * 70)
    print(f"Forecast horizon: {config['model']['forecast_horizon']}")
    print(f"Forecast mean (first 5): {forecast_mean[:5]}")
    print(f"Forecast std (first 5): {forecast_std[:5]}")

    if test_df is not None and len(test_df) >= len(forecast_mean):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Apply SignalPlot's clean defaults
signalplot.apply()

        test_values = test_df[config["data"]["value_col"]].values[: len(forecast_mean)]
        mae = mean_absolute_error(test_values, forecast_mean)
        rmse = np.sqrt(mean_squared_error(test_values, forecast_mean))
        r2 = r2_score(test_values, forecast_mean)

        print(f"\nModel Evaluation (on test set):")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")

    create_visualizations(
        df,
        train_df[config["data"]["value_col"]],
        test_df[config["data"]["value_col"]] if test_df is not None else None,
        forecast_mean,
        forecast_std,
        config,
    )

    print("✓ BSTS forecasting complete")


if __name__ == "__main__":
    main()
