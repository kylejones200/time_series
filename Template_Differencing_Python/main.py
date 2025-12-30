#!/usr/bin/env python3
"""
Differencing for Stationarity
Visualize original series and successive differences (+ADF diagnostics).
"""

import io
from importlib import util
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import signalplot
import numpy as np
import pandas as pd
import requests
import yaml
from statsmodels.tsa.stattools import adfuller

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




def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_remote_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def load_series(config: dict) -> pd.Series:
    if config["data"].get("url"):
        df = fetch_remote_csv(config["data"]["url"])
    else:
        csv_path = Path(__file__).parent.parent / "data" / config["data"]["input_file"]
        df = pd.read_csv(csv_path)

    df[config["data"]["date_col"]] = pd.to_datetime(
        df[config["data"]["date_col"]], errors="coerce"
    )
    df = df.dropna(subset=[config["data"]["date_col"], config["data"]["value_col"]])
    df = df.sort_values(config["data"]["date_col"])
    df = df.set_index(config["data"]["date_col"])

    if config["data"].get("resample_rule"):
        df = df.resample(config["data"]["resample_rule"]).mean().dropna()

    series = pd.to_numeric(df[config["data"]["value_col"]], errors="coerce").dropna()
    series.name = config["data"]["value_col"]
    return series


def adf_summary(series: pd.Series) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    keys = ["ADF Statistic", "p-value", "Lags Used", "Number of Observations"]
    return {k: v for k, v in zip(keys, result[:4])}


def compute_differences(series: pd.Series, max_order: int) -> list[pd.Series]:
    diffs = [series]
    current = series
    for _ in range(max_order):
        current = current.diff().dropna()
        diffs.append(current)
    return diffs


def plot_differences(
    series_list: list[pd.Series], config: dict, adf_results: list[dict]
) -> None:
    figure_cfg = config["plotting"]
    fig, axes = plt.subplots(
        len(series_list), 1, figsize=figure_cfg["figure_size"], sharex=True
    )
    if len(series_list) == 1:
        axes = [axes]

    for idx, (series, ax) in enumerate(zip(series_list, axes)):
                ax.plot(
            series.index,
            series.values,
            color=figure_cfg["colors"][0],
            linewidth=figure_cfg["linewidth"],
            alpha=figure_cfg["alpha"],
        )

        title = config["plot_titles"]["base"] if idx == 0 else f"{idx} order difference"
        info = adf_results[idx]
        ax.set_title(
            f"{title} | ADF: {info['ADF Statistic']:.3f}, p={info['p-value']:.3f}"
        )
        ax.set_ylabel(config["plotting"].get("y_label", "Value"))

    axes[-1].set_xlabel(config["plotting"].get("x_label", "Date"))

    output_path = Path(__file__).parent / "outputs" / "differencing_plot.png"
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def main():
    config = load_config()
    series = load_series(config)

    max_order = config["model"]["max_difference_order"]
    series_list = compute_differences(series, max_order)
    adf_results = [adf_summary(s) for s in series_list]

    metrics_path = Path(__file__).parent / "outputs" / "adf_results.csv"
    pd.DataFrame(adf_results).to_csv(metrics_path, index=False)
    print(f"ADF results saved to {metrics_path}")

    plot_differences(series_list, config, adf_results)


if __name__ == "__main__":
    main()
