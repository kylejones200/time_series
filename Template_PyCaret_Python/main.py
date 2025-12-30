#!/usr/bin/env python3
"""
PyCaret: Low-Code Time Series Forecasting
Automated time series forecasting with minimal code.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from pycaret.time_series import *

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


def setup_pycaret(data, config):
    """Setup PyCaret time series environment."""
    return setup(
        data=data,
        fh=config["model"]["forecast_horizon"],
        session_id=config["model"].get("session_id", 42),
        verbose=False,
    )


def create_and_compare_models(config):
    """Create and compare multiple models."""
    best_model = compare_models(
        include=config["model"].get("models", ["arima", "exp_smooth", "theta"]),
        sort=config["model"].get("sort_metric", "MAE"),
        verbose=False,
    )

    return best_model


def finalize_model(model):
    """Finalize the best model."""
    return finalize_model(model)


def create_visualizations(model, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    plot_model(model, plot="forecast", save=True)

    fig, ax = plt.subplots(figsize=config)

    plot_model(model, plot="forecast", display=False, save=False)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    plt.tight_layout()

    [
        fig.savefig(output_dir / "pycaret_forecast.png", dpi=300, bbox_inches="tight", facecolor="white")
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

    setup_pycaret(data, config)
    best_model = create_and_compare_models(config)
    final_model = finalize_model(best_model)
    create_visualizations(final_model, config)

    print("✓ PyCaret time series forecasting complete")
    print(f"Best model: {type(final_model).__name__}")


if __name__ == "__main__":
    main()
