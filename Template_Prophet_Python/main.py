#!/usr/bin/env python3
"""
Prophet: Facebook's Time Series Forecasting
Automatic forecasting procedure for business time series.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from prophet import Prophet

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


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(data):
    """Prepare data for Prophet (requires 'ds' and 'y' columns)."""
    df = pd.DataFrame({"ds": data.index, "y": data.values})
    return df


def create_prophet_model(config):
    """Create Prophet model with config parameters."""
    model_params = {
        "yearly_seasonality": config["model"].get("yearly_seasonality", "auto"),
        "weekly_seasonality": config["model"].get("weekly_seasonality", "auto"),
        "daily_seasonality": config["model"].get("daily_seasonality", False),
        "seasonality_mode": config["model"].get("seasonality_mode", "additive"),
        "growth": config["model"].get("growth", "linear"),
    }

    [
        model_params.__setitem__(k, v)
        for k, v in config["model"].get("params", {}).items()
    ]

    return Prophet(**model_params)


def fit_and_predict(model, df, config):
    """Fit model and generate predictions."""
    model.fit(df)

    future = model.make_future_dataframe(periods=config["model"]["forecast_horizon"])
    forecast = model.predict(future)

    return forecast


def create_visualizations(model, forecast, df, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=config)

    ax.plot(
        df["ds"],
        df["y"],
        c=config["plotting"]["style"]["colors"]["primary"],
        linewidth=config["plotting"]["style"]["linewidth"],
        alpha=config["plotting"]["style"]["alpha"],
        label="Historical",
    )

    ax.plot(
        forecast["ds"],
        forecast["yhat"],
        c=config["plotting"]["style"]["colors"]["secondary"],
        linewidth=config["plotting"]["style"]["linewidth"],
        label="Forecast",
    )

    [
        ax.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            alpha=0.2,
            color=config["plotting"]["style"]["colors"]["secondary"],
        )
        for _ in [None]
        if "yhat_lower" in forecast.columns
    ]

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    plt.tight_layout()

    [
        fig.savefig(output_dir / "prophet_forecast.png", dpi=300, bbox_inches="tight", facecolor="white")
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

    df = prepare_data(data)
    model = create_prophet_model(config)
    forecast = fit_and_predict(model, df, config)
    create_visualizations(model, forecast, df, config)

    print("✓ Prophet forecasting complete")


if __name__ == "__main__":
    main()
