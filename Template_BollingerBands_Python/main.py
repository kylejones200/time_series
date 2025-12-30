#!/usr/bin/env python3
"""
Bollinger Bands for Time Series Analysis
Technical indicator using moving averages and standard deviations.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util

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
ensure_datetime_index = ts_utils.ensure_datetime_index


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def calculate_bollinger_bands(data, window, num_std):
    """Calculate Bollinger Bands."""
    df = data.copy()
    df["MA"] = df.rolling(window=window).mean()
    df["std"] = df.rolling(window=window).std()
    df["upper"] = df["MA"] + (df["std"] * num_std)
    df["lower"] = df["MA"] - (df["std"] * num_std)
    return df


def main():
    config = load_config()

    df = load_ts_data(
        data_path=Path(__file__).parent.parent / "data" / config["data"]["input_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["value_col"],
    )
    df = ensure_datetime_index(df, time_col=config["data"]["date_col"])

    df_bb = calculate_bollinger_bands(
        df[config["data"]["value_col"]],
        config["model"]["window"],
        config["model"]["num_std"],
    )

    fig, ax = plt.subplots(figsize=tuple(config["plotting"]["figure_size"]))

    ax.plot(
        df.index,
        df[config["data"]["value_col"]].values,
        "k-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label="Price",
    )
    ax.plot(
        df_bb.index,
        df_bb["MA"].values,
        "b-",
        linewidth=config["plotting"]["linewidth"],
        label="Moving Average",
    )
    ax.fill_between(
        df_bb.index,
        df_bb["lower"].values,
        df_bb["upper"].values,
        color="b",
        alpha=0.2,
        label="Bollinger Bands",
    )

    ax.set_title(config["plot_titles"]["bollinger_bands"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    output_path = Path(__file__).parent / "outputs" / "bollinger_bands.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
