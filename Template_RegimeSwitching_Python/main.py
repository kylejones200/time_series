#!/usr/bin/env python3
"""
Regime Switching Models for Time Series
Markov switching models for time series with structural breaks.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

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


def main():
    config = load_config()

    df = load_ts_data(
        data_path=Path(__file__).parent.parent / "data" / config["data"]["input_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["value_col"],
    )
    df = ensure_datetime_index(df, time_col=config["data"]["date_col"])

    data = df[config["data"]["value_col"]].values

    model = MarkovRegression(
        data,
        k_regimes=config["model"]["k_regimes"],
        trend=config["model"]["trend"],
        switching_variance=config["model"]["switching_variance"],
    )

    result = model.fit()
    print("\nMarkov Switching Model Results:")
    print("=" * 70)
    print(result.summary())

    print("\nTransition Matrix:")
    print(result.regime_transition)

    smoothed_probs = result.smoothed_marginal_probabilities

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for ax in axes:
        
    axes[0].plot(
        df.index,
        data,
        "k-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label="Time Series",
    )
    axes[0].set_title(config["plot_titles"]["regime_switching"])
    axes[0].set_ylabel("Value")
    ax.legend()

    for regime in range(config["model"]["k_regimes"]):
        axes[1].plot(
            df.index,
            smoothed_probs[:, regime],
            linewidth=config["plotting"]["linewidth"],
            alpha=config["plotting"]["alpha"],
            label=f"Regime {regime + 1} Probability",
        )
    axes[1].set_title("Smoothed Regime Probabilities")
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Date")
    ax.legend()

    plt.tight_layout()

    output_path = Path(__file__).parent / "outputs" / "regime_switching.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
