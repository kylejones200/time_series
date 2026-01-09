#!/usr/bin/env python3
"""
Econometric Time Series Analysis
Causal inference, policy evaluation, and economic modeling methods.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from linearmodels.panel import PanelOLS


def granger_causality_test(data: pd.DataFrame, x_col: str, y_col: str, max_lag: int, config: dict):
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


def regression_discontinuity(data: pd.DataFrame, date_col: str, value_col: str, cutoff_date: str, config: dict, script_dir: Path):
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
    
    # Create visualization
    fig, ax = plt.subplots(figsize=tuple(config.get("plotting", {}).get("figure_size", [12, 6])))
    
    pre_treatment = df[df["treatment"] == 0]
    post_treatment = df[df["treatment"] == 1]
    
    ax.scatter(pre_treatment["time_from_cutoff"], pre_treatment[value_col], alpha=0.6, s=20, label="Pre-treatment")
    ax.scatter(post_treatment["time_from_cutoff"], post_treatment[value_col], alpha=0.6, s=20, label="Post-treatment")
    ax.axvline(0, color="r", linestyle="--", lw=2, label="Cutoff")
    ax.set_xlabel("Days from Cutoff")
    ax.set_ylabel(value_col)
    ax.set_title(f"Regression Discontinuity Design (Treatment Effect: {treatment_effect:.4f})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / "rdd_analysis.png", dpi=300)
    print(f"Plot saved to: {output_dir / 'rdd_analysis.png'}")
    plt.close(fig)


def panel_regression(data: pd.DataFrame, config: dict):
    """Perform panel regression analysis."""
    print("\nPanel Regression Analysis")
    print("=" * 70)
    
    entity_col = config["model"]["entity_col"]
    time_col = config["model"]["time_col"]
    y_col = config["model"]["y_col"]
    x_cols = config["model"]["x_cols"]
    
    data = data.reset_index()
    data = data.set_index([entity_col, time_col])
    
    model = PanelOLS.from_formula(
        f"{y_col} ~ {' + '.join(x_cols)}",
        data=data,
        entity_effects=config["model"].get("entity_effects", True),
        time_effects=config["model"].get("time_effects", True),
    )
    
    result = model.fit(cov_type="clustered", cluster_entity=True)
    print(result.summary)


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data - Econometrics may require multi-series data
    data_path = script_dir.parent / "data" / config["data"]["input_file"]
    df = pd.read_csv(data_path, encoding="utf-8")
    
    # Process date column
    date_col = config["data"].get("date_col", "date")
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    
    print(f"Loaded {len(df)} data points")
    print(f"Columns: {list(df.columns)}")
    
    # Granger causality test if configured
    if config["model"].get("granger_test", {}).get("enabled", False):
        x_col = config["model"]["granger_test"]["x_col"]
        y_col = config["model"]["granger_test"]["y_col"]
        max_lag = config["model"]["granger_test"]["max_lag"]
        granger_causality_test(df, x_col, y_col, max_lag, config)
    
    # Regression discontinuity if configured
    if config["model"].get("rdd", {}).get("enabled", False):
        value_col = config["model"]["rdd"]["value_col"]
        cutoff_date = config["model"]["rdd"]["cutoff_date"]
        regression_discontinuity(df, date_col, value_col, cutoff_date, config, script_dir)
    
    # Panel regression if configured
    if config["model"].get("panel_regression", {}).get("enabled", False):
        panel_regression(df, config)
    
    print("\n Econometric analysis complete")


if __name__ == "__main__":
    main()
