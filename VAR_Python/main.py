#!/usr/bin/env python3
"""
Vector Autoregression (VAR) for Multivariate Time Series
VAR modeling for multiple interdependent time series using statsmodels.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.evaluator import Evaluator

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_multivariate_data(config: dict) -> pd.DataFrame:
    """Load multivariate time series data."""
    input_file = config["data"]["input_file"]
    data_path = Path(input_file)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / "data" / input_file
    df = pd.read_csv(data_path)
    df[config["data"]["date_col"]] = pd.to_datetime(df[config["data"]["date_col"]])
    df = df.set_index(config["data"]["date_col"]).sort_index()
    requested_cols = list(config["data"]["value_cols"])
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns available for VAR")
    base_col = numeric_cols[0]
    for i, col in enumerate(requested_cols):
        if col not in df.columns:
            if i == 0:
                df[col] = df[base_col]
            else:
                df[col] = df[base_col].shift(i)
    df = df[requested_cols].dropna()
    return df


def check_stationarity(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Check stationarity of all series and apply differencing if needed."""
    stationary_data = data.copy()
    differenced = False
    
    for col in data.columns:
        result = adfuller(data[col].dropna())
        if result[1] > 0.05:
            print(f"{col}: Non-stationary (p-value={result[1]:.4f}), differencing...")
            differenced = True
        else:
            print(f"{col}: Stationary (p-value={result[1]:.4f})")
    
    if differenced or config["model"]["force_differencing"]:
        stationary_data = data.diff().dropna()
        print("\nApplied first-order differencing")
    
    return stationary_data


def fit_var_model(train_data: pd.DataFrame, config: dict) -> tuple:
    """Fit VAR model with optimal lag selection."""
    model = VAR(train_data)
    lag_order = model.select_order(maxlags=config["model"]["max_lags"])
    
    optimal_lag = lag_order.aic if config["model"]["use_aic"] else lag_order.bic
    print(f"\nOptimal lag order: {optimal_lag}")
    
    fitted_model = model.fit(optimal_lag)
    print(f"\nVAR({optimal_lag}) Model Summary:")
    print(fitted_model.summary())
    
    return fitted_model, optimal_lag


def granger_causality_test(model, data: pd.DataFrame, config: dict):
    """Perform Granger causality tests."""
    if config["model"]["granger_test"]:
        print("\nGranger Causality Tests:")
        for col1 in data.columns:
            for col2 in data.columns:
                if col1 != col2:
                    print(f"\nTesting if {col2} Granger-causes {col1}:")
                    try:
                        gc_result = grangercausalitytests(
                            data[[col1, col2]],
                            maxlag=config["model"]["granger_maxlag"],
                            verbose=False,
                        )
                        p_values = [
                            gc_result[lag][0]["ssr_ftest"][1]
                            for lag in range(1, config["model"]["granger_maxlag"] + 1)
                        ]
                        min_p = min(p_values)
                        print(f"  Minimum p-value: {min_p:.4f}", end="")
                        if min_p < 0.05:
                            print(" → Significant Granger causality")
                        else:
                            print(" → No significant Granger causality")
                    except Exception as e:
                        print(f"  Error: {e}")


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load multivariate data
    df = load_multivariate_data(config)
    print(f"Loaded {len(df)} data points with {len(config['data']['value_cols'])} variables")
    
    # Check stationarity
    print("\nStationarity Tests:")
    stationary_df = check_stationarity(df[config["data"]["value_cols"]], config)
    
    # Split train/test - for multivariate, we'll split by index
    test_size = config.get("data", {}).get("test_size", 0.2)
    split_idx = int(len(stationary_df) * (1 - test_size))
    train_df = stationary_df.iloc[:split_idx]
    test_df = stationary_df.iloc[split_idx:]
    
    print(f"\nTrain: {len(train_df)} points, Test: {len(test_df)} points")
    
    # Fit VAR model
    model, optimal_lag = fit_var_model(train_df, config)
    
    # Granger causality tests
    if config["model"]["granger_test"]:
        granger_causality_test(model, train_df, config)
    
    # Residual diagnostics
    residuals = model.resid
    print("\nDurbin-Watson Test (should be close to 2):")
    for col in residuals.columns:
        dw_stat = durbin_watson(residuals[col])
        print(f"  {col}: {dw_stat:.2f}")
    
    # Generate forecast
    forecast = model.forecast(train_df.values[-optimal_lag:], steps=len(test_df))
    forecast_df = pd.DataFrame(forecast, index=test_df.index, columns=test_df.columns)
    
    # Undo differencing if needed
    if config["model"]["force_differencing"] or not stationary_df.equals(df[config["data"]["value_cols"]].dropna()):
        forecast_actual = (
            forecast_df.cumsum()
            + df[config["data"]["value_cols"]].iloc[-len(test_df) - 1].values
        )
        forecast_actual = pd.DataFrame(
            forecast_actual, index=test_df.index, columns=test_df.columns
        )
    else:
        forecast_actual = forecast_df
    
    # Evaluate
    metrics = {}
    for col in test_df.columns:
        if col in df.columns:
            actual = df[col].iloc[-len(test_df):].values
            pred = forecast_actual[col].values
            metrics[col] = {
                "MAE": mean_absolute_error(actual, pred),
                "RMSE": np.sqrt(mean_squared_error(actual, pred)),
                "R²": r2_score(actual, pred),
            }
    
    print("\nModel Evaluation:")
    for col, m in metrics.items():
        print(f"\n{col}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")
    
    # Create visualizations
    n_vars = len(config["data"]["value_cols"])
    fig, axes = plt.subplots(n_vars, 1, figsize=(15, 5 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]
    
    for i, col in enumerate(config["data"]["value_cols"]):
        axes[i].plot(
            df.index[-100:],
            df[col].values[-100:],
            "k-",
            linewidth=2,
            alpha=0.7,
            label="Historical",
        )
        if col in test_df.columns:
            axes[i].plot(
                test_df.index,
                df[col].iloc[-len(test_df):].values,
                "g-",
                linewidth=2,
                alpha=0.7,
                label="Actual (Test)",
            )
        if col in forecast_actual.columns:
            axes[i].plot(
                forecast_actual.index,
                forecast_actual[col].values,
                "r--",
                linewidth=2,
                label="Forecast",
            )
        
        axes[i].set_title(f"VAR Forecast: {col}")
        axes[i].set_ylabel("Value")
        axes[i].legend(loc="best")
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    
    # Save plot using consolidated utility
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    dpi = config.get("plotting", {}).get("dpi", 300)
    save_plot(fig, output_dir / "var_forecast.png", dpi=dpi)
    print(f"\nPlot saved to: {output_dir / 'var_forecast.png'}")
    
    print("\n VAR analysis complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
