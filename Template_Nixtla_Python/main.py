#!/usr/bin/env python3
"""
Nixtla: StatsForecast
Fast statistical forecasting with Nixtla's StatsForecast library.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from statsforecast import StatsForecast
from statsforecast.models import (

# Apply SignalPlot's clean defaults
signalplot.apply()
    AutoARIMA,
    AutoETS,
    AutoTheta,
    AutoCES,
    DynamicOptimizedTheta,
    SeasonalNaive,
)


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


def prepare_data(data, config):
    """Prepare data for StatsForecast (requires 'ds' and 'y' columns, 'unique_id' for multiple series)."""
    df = pd.DataFrame(
        {
            "unique_id": config["data"].get("unique_id", "series1"),
            "ds": data.index,
            "y": data.values,
        }
    )
    return df


def create_models(config):
    """Create list of models for StatsForecast."""
    model_map = {
        "AutoARIMA": AutoARIMA,
        "AutoETS": AutoETS,
        "AutoTheta": AutoTheta,
        "AutoCES": AutoCES,
        "DynamicOptimizedTheta": DynamicOptimizedTheta,
        "SeasonalNaive": SeasonalNaive,
    }

    model_names = config["model"].get("models", ["AutoARIMA"])
    return [model_map[name]() for name in model_names]


def fit_and_predict(models, df, config):
    """Fit models and generate predictions."""
    sf = StatsForecast(
        models=models,
        freq=config["model"].get("freq", "D"),
        n_jobs=config["model"].get("n_jobs", 1),
    )

    sf.fit(df)

    forecast_horizon = config["model"]["forecast_horizon"]
    forecasts = sf.predict(h=forecast_horizon)

    return sf, forecasts


def create_visualizations(df, forecasts, config):
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

    model_cols = [col for col in forecasts.columns if col not in ["ds", "unique_id"]]

    [
        [
            ax.plot(
                forecasts["ds"],
                forecasts[col],
                linewidth=config["plotting"]["style"]["linewidth"],
                label=col,
            )
            for col in model_cols[:3]
        ]
        for _ in [None]
    ]

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    plt.tight_layout()

    [
        fig.savefig(output_dir / "nixtla_forecast.png", dpi=300, bbox_inches="tight", facecolor="white")
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

    df = prepare_data(data, config)
    models = create_models(config)
    sf, forecasts = fit_and_predict(models, df, config)
    create_visualizations(df, forecasts, config)

    print("✓ Nixtla StatsForecast complete")


if __name__ == "__main__":
    main()
