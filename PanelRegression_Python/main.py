#!/usr/bin/env python3
"""Panel Regression with Driscoll-Kraay Standard Errors for panel time series data."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import linearmodels
try:
    from linearmodels.panel import PanelOLS
    LINEARMODELS_AVAILABLE = True
except ImportError:
    LINEARMODELS_AVAILABLE = False
    warnings.warn("linearmodels not available. Install with: pip install linearmodels")

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import consolidated utilities
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

warnings.filterwarnings("ignore")


def load_panel_data(config: dict) -> pd.DataFrame:
    """
    Load panel data (multi-index: entity, date).
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    pd.DataFrame
        Panel data with multi-index (entity, date)
    """
    data_config = config["data"]
    repo_root = Path(__file__).parent.parent
    data_path = repo_root / data_config["input_file"]
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path, encoding="utf-8")
    
    # Parse date column
    date_col = data_config.get("date_column", "date")
    entity_col = data_config.get("entity_column", "entity_id")
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Set multi-index
    df = df.set_index([entity_col, date_col])
    df = df.sort_index()
    
    return df


def fit_panel_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: list,
    config: dict,
) -> dict:
    """
    Fit panel regression models with different standard error types.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Panel data
    dependent_var : str
        Dependent variable column name
    independent_vars : list
        List of independent variable column names
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    dict
        Dictionary with fitted models and standard errors
    """
    if not LINEARMODELS_AVAILABLE:
        raise ImportError("linearmodels is required for panel regression")
    
    model_config = config.get("model", {})
    
    # Prepare data
    y = data[dependent_var]
    X = data[independent_vars]
    X = sm.add_constant(X)  # Add intercept
    
    # Drop missing values
    valid_idx = ~(y.isna() | X.isna().any(axis=1))
    y = y[valid_idx]
    X = X[valid_idx]
    
    # Entity and time effects
    entity_effects = model_config.get("entity_effects", True)
    time_effects = model_config.get("time_effects", False)
    
    results = {}
    
    # Driscoll-Kraay standard errors
    if model_config.get("driscoll_kraay", True):
        print("Fitting panel model with Driscoll-Kraay standard errors...")
        dk_model = PanelOLS(
            y,
            X,
            entity_effects=entity_effects,
            time_effects=time_effects,
        ).fit(
            cov_type="kernel",
            kernel="bartlett",
            bandwidth=model_config.get("bandwidth", 3),
        )
        results["driscoll_kraay"] = {
            "model": dk_model,
            "std_errors": dk_model.std_errors.values,
            "params": dk_model.params.values,
        }
        print(f"Driscoll-Kraay SEs: {results['driscoll_kraay']['std_errors']}")
    
    # Clustered standard errors
    if model_config.get("clustered", True):
        print("\nFitting panel model with clustered standard errors...")
        cluster_model = PanelOLS(
            y,
            X,
            entity_effects=entity_effects,
            time_effects=time_effects,
        ).fit(
            cov_type="clustered",
            cluster_entity=model_config.get("cluster_entity", True),
            cluster_time=model_config.get("cluster_time", False),
        )
        results["clustered"] = {
            "model": cluster_model,
            "std_errors": cluster_model.std_errors.values,
            "params": cluster_model.params.values,
        }
        print(f"Clustered SEs: {results['clustered']['std_errors']}")
    
    # Robust standard errors
    if model_config.get("robust", False):
        print("\nFitting panel model with robust standard errors...")
        robust_model = PanelOLS(
            y,
            X,
            entity_effects=entity_effects,
            time_effects=time_effects,
        ).fit(cov_type="robust")
        results["robust"] = {
            "model": robust_model,
            "std_errors": robust_model.std_errors.values,
            "params": robust_model.params.values,
        }
        print(f"Robust SEs: {results['robust']['std_errors']}")
    
    return results


def create_panel_visualization(
    data: pd.DataFrame,
    dependent_var: str,
    results: dict,
    config: dict,
) -> plt.Figure:
    """
    Create visualization of panel regression results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Panel data
    dependent_var : str
        Dependent variable name
    results : dict
        Dictionary with fitted models
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    plt.Figure
        Figure with visualizations
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sample entities over time
    entity_level = data.index.names[0]
    sample_entities = data.index.get_level_values(entity_level).unique()[:5]
    
    for entity in sample_entities:
        entity_data = data.xs(entity, level=entity_level)
        axes[0, 0].plot(
            entity_data.index,
            entity_data[dependent_var],
            linewidth=1,
            alpha=0.7,
            label=f"Entity {entity}",
        )
    
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel(dependent_var)
    axes[0, 0].set_title("Sample Entities Over Time")
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)
    
    # Plot 2: Standard error comparison
    if "driscoll_kraay" in results and "clustered" in results:
        dk_se = results["driscoll_kraay"]["std_errors"]
        cluster_se = results["clustered"]["std_errors"]
        labels = ["Intercept"] + config["model"].get("independent_vars", ["var1"])
        
        x_pos = np.arange(len(labels))
        width = 0.35
        
        axes[0, 1].bar(
            x_pos - width / 2,
            dk_se,
            width,
            color="gray",
            alpha=0.7,
            label="Driscoll-Kraay SEs",
        )
        axes[0, 1].bar(
            x_pos + width / 2,
            cluster_se,
            width,
            color="blue",
            alpha=0.5,
            label="Clustered SEs",
        )
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(labels)
        axes[0, 1].set_ylabel("Standard Error")
        axes[0, 1].set_title("Standard Error Comparison")
        axes[0, 1].legend(frameon=False)
        axes[0, 1].spines["top"].set_visible(False)
        axes[0, 1].spines["right"].set_visible(False)
        axes[0, 1].grid(True, alpha=0.3, axis="y")
    
    # Plot 3: Distribution of dependent variable
    axes[1, 0].hist(data[dependent_var].values, bins=50, alpha=0.7, edgecolor="black")
    axes[1, 0].set_xlabel(dependent_var)
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title(f"Distribution of {dependent_var}")
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    
    # Plot 4: Residuals (if available)
    if "driscoll_kraay" in results:
        model = results["driscoll_kraay"]["model"]
        residuals = model.resids.values.flatten()
        axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.5, s=1)
        axes[1, 1].axhline(y=0, color="r", linestyle="--", linewidth=1)
        axes[1, 1].set_xlabel("Observation")
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].set_title("Residuals Plot")
        axes[1, 1].spines["top"].set_visible(False)
        axes[1, 1].spines["right"].set_visible(False)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    if not LINEARMODELS_AVAILABLE:
        print("ERROR: linearmodels is not installed.")
        print("Install with: pip install linearmodels")
        sys.exit(1)
    
    script_dir = Path(__file__).parent
    config = load_config(script_dir / "config.yaml")
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    
    # Load panel data
    print("Loading panel data...")
    data = load_panel_data(config)
    
    data_config = config["data"]
    model_config = config.get("model", {})
    
    dependent_var = data_config.get("dependent_variable", "value")
    independent_vars = model_config.get("independent_vars", ["days"])
    
    print(f"\nPanel data shape: {data.shape}")
    print(f"Number of entities: {data.index.get_level_values(0).nunique()}")
    print(f"Date range: {data.index.get_level_values(1).min()} to {data.index.get_level_values(1).max()}")
    
    # Fit panel models
    results = fit_panel_model(
        data,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        config=config,
    )
    
    # Print model summaries
    print("\n" + "=" * 70)
    print("MODEL SUMMARIES")
    print("=" * 70)
    
    for se_type, result in results.items():
        print(f"\n{se_type.upper().replace('_', ' ')} Model:")
        print(result["model"].summary)
    
    # Create visualization
    print("\nCreating visualization...")
    fig = create_panel_visualization(data, dependent_var, results, config)
    
    plot_path = output_dir / config["output"].get("plot_file", "panel_regression.png")
    save_plot(fig, plot_path, dpi=config["output"].get("dpi", 300))
    print(f"Plot saved to: {plot_path}")
    
    # Save results
    results_df = pd.DataFrame({
        "coefficient": ["Intercept"] + independent_vars,
    })
    
    for se_type, result in results.items():
        results_df[f"{se_type}_param"] = result["params"]
        results_df[f"{se_type}_std_error"] = result["std_errors"]
    
    csv_path = output_dir / config["output"].get("results_file", "panel_regression_results.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Results saved to: {csv_path}")
    
    print("\n Panel regression analysis complete")


if __name__ == "__main__":
    main()

