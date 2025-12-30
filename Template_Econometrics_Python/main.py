#!/usr/bin/env python3
"""
Econometric Time Series Analysis
Causal inference, policy evaluation, and economic modeling methods.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

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
split_ts = ts_utils.split_ts


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def granger_causality_test(data, x_col, y_col, max_lag, config):
    """Perform Granger causality test."""
    print(f"\nGranger Causality Test: Does {x_col} Granger-cause {y_col}?")
    print("=" * 70)

    test_data = data[[y_col, x_col]].dropna()

    try:
        gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

        p_values = []
        for lag in range(1, max_lag + 1):
            if lag in gc_result:
                p_value = gc_result[lag][0]["ssr_ftest"][1]
                p_values.append((lag, p_value))
                print(f"Lag {lag}: p-value = {p_value:.4f}", end="")
                if p_value < 0.05:
                    print(" → Significant Granger causality")
                else:
                    print(" → No significant Granger causality")

        min_p = min(p_values, key=lambda x: x[1])
        print(f"\nMinimum p-value: {min_p[1]:.4f} at lag {min_p[0]}")

        return gc_result, min_p
    except Exception as e:
        print(f"Error in Granger causality test: {e}")
        return None, None


def regression_discontinuity(data, date_col, value_col, cutoff_date, config):
    """Perform Regression Discontinuity Design (RDD) analysis."""
    print(f"\nRegression Discontinuity Design Analysis")
    print("=" * 70)

    df = data.copy()
    df = df.reset_index()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = pd.to_datetime(cutoff_date)

    df["time_from_cutoff"] = (df[date_col] - cutoff).dt.days
    df["treatment"] = (df["time_from_cutoff"] >= 0).astype(int)
    df["interaction"] = df["time_from_cutoff"] * df["treatment"]

    model = ols(f"{value_col} ~ time_from_cutoff * treatment", data=df).fit()

    print(model.summary())

    treatment_effect = model.params["treatment"]
    print(f"\nTreatment Effect: {treatment_effect:.4f}")

    fig, ax = plt.subplots(figsize=tuple(config["plotting"]["figure_size"]))
    
    pre_treatment = df[df["treatment"] == 0]
    post_treatment = df[df["treatment"] == 1]

    ax.scatter(
        pre_treatment["time_from_cutoff"],
        pre_treatment[value_col],
        alpha=config["plotting"]["alpha"],
        s=config["plotting"]["markersize"] * 10,
        label="Pre-treatment",
        c="b",
    )
    ax.scatter(
        post_treatment["time_from_cutoff"],
        post_treatment[value_col],
        alpha=config["plotting"]["alpha"],
        s=config["plotting"]["markersize"] * 10,
        label="Post-treatment",
        c="r",
    )
    ax.axvline(x=0, color="k", linestyle="--", linewidth=2, label="Cutoff")

    ax.set_title(config["plot_titles"]["rdd_analysis"])
    ax.set_xlabel("Time from Cutoff (days)")
    ax.set_ylabel("Value")
    ax.legend()

    output_path = Path(__file__).parent / "outputs" / "rdd_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return model


def ols_regression(data, y_col, x_cols, config):
    """Perform OLS regression analysis."""
    print(f"\nOLS Regression Analysis")
    print("=" * 70)

    df = data[[y_col] + x_cols].dropna()

    y = df[y_col]
    X = df[x_cols]
    X = sm.add_constant(X)

    model = OLS(y, X).fit()
    print(model.summary())

    return model


def var_analysis(data, value_cols, config):
    """Perform Vector Autoregression (VAR) analysis."""
    print(f"\nVector Autoregression (VAR) Analysis")
    print("=" * 70)

    df = data[value_cols].dropna()

    for col in value_cols:
        result = adfuller(df[col])
        print(f"{col} ADF p-value: {result[1]:.4f}", end="")
        if result[1] > 0.05:
            print(" → Non-stationary, differencing required")
        else:
            print(" → Stationary")

    df_diff = df.diff().dropna()

    model = VAR(df_diff)
    lag_order = model.select_order(maxlags=config["model"]["var_max_lags"])
    optimal_lag = lag_order.aic

    print(f"\nOptimal lag order (AIC): {optimal_lag}")

    fitted_model = model.fit(optimal_lag)
    print(fitted_model.summary())

    return fitted_model, optimal_lag


def panel_regression(data, config):
    """Perform panel regression with Driscoll-Kraay and clustered standard errors."""
    print(f"\nPanel Regression (Fixed Effects)")
    print("=" * 70)

    model_cfg = config["model"]
    id_col = model_cfg["panel_id_col"]
    time_col = model_cfg["panel_time_col"]
    y_col = model_cfg["panel_y_col"]
    x_cols = model_cfg["panel_x_cols"]

    subset_cols = [id_col, time_col, y_col] + x_cols
    df = data[subset_cols].dropna().copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index([id_col, time_col]).sort_index()

    if df.empty:
        raise ValueError(
            "Panel regression requires non-empty data after dropping NA values."
        )

    y = df[y_col]
    X = sm.add_constant(df[x_cols])

    kernel = model_cfg.get("panel_kernel", "bartlett")
    bandwidth = model_cfg.get("panel_bandwidth", 3)
    entity_effects = model_cfg.get("panel_entity_effects", True)

    dk_model = PanelOLS(y, X, entity_effects=entity_effects).fit(
        cov_type="kernel",
        kernel=kernel,
        bandwidth=bandwidth,
    )
    clustered_model = PanelOLS(y, X, entity_effects=entity_effects).fit(
        cov_type="clustered",
        cluster_entity=True,
    )

    print("Driscoll-Kraay standard errors:")
    print(dk_model.std_errors)
    print("\nClustered (entity) standard errors:")
    print(clustered_model.std_errors)

    sample_units = df.index.get_level_values(0).unique()[
        : model_cfg.get("panel_plot_units", 5)
    ]
    fig, ax = plt.subplots(figsize=tuple(config["plotting"]["figure_size"]))
    
    for unit in sample_units:
        unit_data = df.xs(unit, level=0)
        ax.plot(
            unit_data.index,
            unit_data[y_col],
            linewidth=config["plotting"]["linewidth"],
            alpha=config["plotting"]["alpha"],
            label=str(unit),
        )

    ax.set_title(config["plot_titles"]["panel_analysis"])
    ax.set_xlabel(time_col)
    ax.set_ylabel(y_col)
    ax.legend()

    output_path = Path(__file__).parent / "outputs" / "panel_sample_units.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return {
        "driscoll_kraay": dk_model,
        "clustered": clustered_model,
    }


def main():
    config = load_config()

    data_path = Path(__file__).parent.parent / "data" / config["data"]["input_file"]
    df = pd.read_csv(data_path)
    df[config["data"]["date_col"]] = pd.to_datetime(df[config["data"]["date_col"]])
    df = df.sort_values(config["data"]["date_col"])
    ts_df = df.set_index(config["data"]["date_col"]).sort_index()

    method = config["model"]["method"]

    method_map = {
        "granger": lambda: granger_causality_test(
            ts_df,
            config["model"]["x_col"],
            config["model"]["y_col"],
            config["model"]["max_lag"],
            config,
        ),
        "rdd": lambda: regression_discontinuity(
            ts_df,
            config["data"]["date_col"],
            config["data"]["value_col"],
            config["model"]["cutoff_date"],
            config,
        ),
        "ols": lambda: ols_regression(
            ts_df, config["model"]["y_col"], config["model"]["x_cols"], config
        ),
        "var": lambda: var_analysis(ts_df, config["data"]["value_cols"], config),
        "panel": lambda: panel_regression(df, config),
    }

    result = method_map.get(method, lambda: print(f"Unknown method: {method}"))()
    print(f"\n✓ Econometric analysis ({method}) complete")


if __name__ == "__main__":
    main()
