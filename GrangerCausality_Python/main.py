#!/usr/bin/env python3
"""Granger Causality testing and multivariate forecasting using leading indicators."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import consolidated utilities
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.evaluator import Evaluator

warnings.filterwarnings("ignore")


def load_multivariate_data(
    config: dict,
    script_dir: Path,
) -> pd.DataFrame:
    """
    Load multivariate time series data.
    
    Supports:
    - Single CSV with multiple value columns
    - Multiple CSV files (one per series)
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    script_dir : Path
        Script directory for path resolution
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and value columns
    """
    data_config = config["data"]
    repo_root = script_dir.parent
    
    # Check if single file with multiple columns
    if "input_file" in data_config:
        first_file = repo_root / data_config["input_file"]
        
        # If single file with multiple columns (preferred for Granger causality)
        if "value_columns" in data_config:
            # Load full CSV
            df_full = pd.read_csv(first_file, encoding="utf-8")
            date_col = data_config.get("date_column", "date")
            df_full[date_col] = pd.to_datetime(df_full[date_col], errors="coerce")
            df_full = df_full.dropna(subset=[date_col])
            df_full = df_full.set_index(date_col).sort_index()
            
            # Extract specified columns
            value_cols = data_config["value_columns"]
            series_names = data_config.get("series_names", value_cols)
            df = df_full[value_cols].rename(columns=dict(zip(value_cols, series_names)))
        
        # If multiple files specified
        elif "input_files" in data_config:
            from src import load_time_series
            series_names = data_config.get("series_names", [f"series{i+1}" for i in range(len(data_config["input_files"]))])
            dfs = []
            for i, file_path in enumerate(data_config["input_files"]):
                file_path = repo_root / file_path
                series = load_time_series(
                    str(file_path),
                    date_column=data_config.get("date_column", "date"),
                    value_column=data_config.get("value_column", "value"),
                )
                dfs.append(pd.DataFrame({series_names[i]: series}))
            # Merge on index
            df = pd.concat(dfs, axis=1)
        
        # Single file, single column (fallback - not ideal for Granger)
        else:
            from src import load_time_series
            first_series = load_time_series(
                str(first_file),
                date_column=data_config.get("date_column", "date"),
                value_column=data_config.get("value_column", "value"),
            )
            df = pd.DataFrame({data_config.get("series_names", ["series1"])[0]: first_series})
        
        return df.dropna()
    
    else:
        raise ValueError("Must specify either 'input_file' or 'input_files' in data config")


def test_stationarity(series: pd.Series, name: str) -> bool:
    """
    Test if time series is stationary using Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    name : str
        Series name for reporting
    
    Returns:
    --------
    bool
        True if stationary (p < 0.05)
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    is_stationary = p_value < 0.05
    
    print(f"  {name}: p-value = {p_value:.4f} {'(stationary)' if is_stationary else '(non-stationary)'}")
    
    return is_stationary


def test_granger_causality(
    df: pd.DataFrame,
    target_col: str,
    predictor_col: str,
    max_lag: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Test if predictor_col Granger-causes target_col.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with both series
    target_col : str
        Target series name
    predictor_col : str
        Predictor series name
    max_lag : int
        Maximum lag to test
    verbose : bool
        Whether to print detailed results
    
    Returns:
    --------
    dict
        Results dictionary with p-values and causality decision
    """
    # Prepare data for Granger test (order: [target, predictor])
    test_data = df[[target_col, predictor_col]].dropna()
    
    if len(test_data) < max_lag + 10:
        raise ValueError(f"Insufficient data: need at least {max_lag + 10} observations")
    
    if verbose:
        print(f"\nTesting if '{predictor_col}' Granger-causes '{target_col}':")
        print("=" * 60)
    
    # Run Granger causality test
    gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=verbose)
    
    # Extract p-values for each lag
    p_values = {}
    min_p = 1.0
    best_lag = None
    
    for lag in range(1, max_lag + 1):
        if lag in gc_result:
            # Extract p-value from F-test
            p_value = gc_result[lag][0]["ssr_ftest"][1]
            p_values[lag] = p_value
            if p_value < min_p:
                min_p = p_value
                best_lag = lag
    
    # Determine causality (p < 0.05 indicates causality)
    is_causal = min_p < 0.05
    
    results = {
        "p_values": p_values,
        "min_p_value": min_p,
        "best_lag": best_lag,
        "is_causal": is_causal,
        "interpretation": f"{predictor_col} {'DOES' if is_causal else 'DOES NOT'} Granger-cause {target_col}",
    }
    
    if verbose:
        print(f"\nSummary:")
        print(f"  Minimum p-value: {min_p:.4f} (lag {best_lag})")
        print(f"  Causality: {results['interpretation']}")
        if is_causal:
            print(f"  → Use {predictor_col} as leading indicator for forecasting {target_col}")
    
    return results


def fit_multivariate_regression(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
    lag: int = 1,
) -> tuple[sm.OLS, pd.Series]:
    """
    Fit multivariate regression model with lagged predictors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with all series
    target_col : str
        Target series name
    predictor_cols : list[str]
        List of predictor series names
    lag : int
        Lag to use for predictors
    
    Returns:
    --------
    tuple
        (fitted_model, predictions)
    """
    # Create lagged features
    X_data = {}
    for col in predictor_cols:
        X_data[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    # Create feature matrix
    X = pd.DataFrame(X_data, index=df.index)
    
    # Align target and features
    aligned = pd.concat([df[[target_col]], X], axis=1).dropna()
    y = aligned[target_col]
    X = aligned[X.columns]
    
    # Add constant term
    X = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    
    # Generate predictions
    predictions = pd.Series(model.predict(X), index=aligned.index)
    
    return model, predictions


def forecast_multivariate(
    model: sm.OLS,
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
    lag: int,
    forecast_horizon: int,
) -> pd.Series:
    """
    Generate forecasts using multivariate regression.
    
    Parameters:
    -----------
    model : sm.OLS
        Fitted regression model
    df : pd.DataFrame
        Historical data
    target_col : str
        Target series name
    predictor_cols : list[str]
        Predictor series names
    lag : int
        Lag used in model
    forecast_horizon : int
        Number of steps to forecast
    
    Returns:
    --------
    pd.Series
        Forecast series with datetime index
    """
    # For simplicity, use last known values of predictors
    # In practice, you'd need forecasts of predictors too
    last_values = {}
    for col in predictor_cols:
        last_values[f"{col}_lag{lag}"] = df[col].iloc[-lag]
    
    # Create forecast index
    last_date = df.index[-1]
    freq = pd.infer_freq(df.index) or "D"
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=freq,
    )
    
    # Generate forecasts (using last known predictor values)
    # This is a simplified approach - in practice, you'd forecast predictors too
    forecasts = []
    
    # Build feature names that match model
    feature_names = []
    if "const" in model.params.index:
        feature_names.append("const")
    for col in predictor_cols:
        feature_names.append(f"{col}_lag{lag}")
    
    for _ in range(forecast_horizon):
        # Create feature vector matching model parameters
        X_forecast_data = {}
        if "const" in model.params.index:
            X_forecast_data["const"] = [1.0]
        for col in predictor_cols:
            X_forecast_data[f"{col}_lag{lag}"] = [last_values[f"{col}_lag{lag}"]]
        
        X_forecast = pd.DataFrame(X_forecast_data, index=[0])
        # Ensure columns match model params exactly
        X_forecast = X_forecast.reindex(columns=model.params.index, fill_value=0)
        forecast = model.predict(X_forecast).iloc[0]
        forecasts.append(forecast)
    
    return pd.Series(forecasts, index=forecast_dates)


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    config = load_config(script_dir / "config.yaml")
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    
    # Load multivariate data
    print("Loading multivariate time series data...")
    df = load_multivariate_data(config, script_dir)
    print(f"Loaded {len(df)} observations with {len(df.columns)} series")
    print(f"Series: {', '.join(df.columns)}")
    
    # Get series names from config
    series_names = config["data"].get("series_names", list(df.columns))
    if len(series_names) < 2:
        raise ValueError("Need at least 2 series for Granger causality testing")
    
    target_col = series_names[0]
    predictor_col = series_names[1]
    
    # Test stationarity (recommended for Granger test)
    print("\nTesting stationarity (ADF test):")
    target_stationary = test_stationarity(df[target_col], target_col)
    predictor_stationary = test_stationarity(df[predictor_col], predictor_col)
    
    if not target_stationary or not predictor_stationary:
        print("\nWarning: Non-stationary series detected.")
        print("Consider differencing before Granger causality test.")
    
    # Test Granger causality
    max_lag = config.get("granger", {}).get("max_lag", 5)
    gc_results = test_granger_causality(
        df,
        target_col=target_col,
        predictor_col=predictor_col,
        max_lag=max_lag,
        verbose=True,
    )
    
    # Correlation analysis
    correlation = df[[target_col, predictor_col]].corr().iloc[0, 1]
    print(f"\nCorrelation between {target_col} and {predictor_col}: {correlation:.4f}")
    
    # Split data for evaluation
    evaluator = Evaluator(test_size=config["evaluation"].get("test_size", 0.2))
    train_df = df.iloc[:int(len(df) * (1 - evaluator.test_size))]
    test_df = df.iloc[int(len(df) * (1 - evaluator.test_size)):]
    
    # Fit multivariate regression if causality found
    if gc_results["is_causal"]:
        lag = gc_results["best_lag"]
        print(f"\nFitting multivariate regression model (lag={lag})...")
        model, train_pred = fit_multivariate_regression(
            train_df,
            target_col=target_col,
            predictor_cols=[predictor_col],
            lag=lag,
        )
        
        print(f"\nRegression Results:")
        print(model.summary())
        
        # Evaluate on test set
        test_model, test_pred = fit_multivariate_regression(
            test_df,
            target_col=target_col,
            predictor_cols=[predictor_col],
            lag=lag,
        )
        
        # Calculate metrics
        test_actual = test_df[target_col].iloc[lag:]
        test_pred_aligned = test_pred[test_pred.index.isin(test_actual.index)]
        test_actual_aligned = test_actual[test_actual.index.isin(test_pred_aligned.index)]
        
        if len(test_pred_aligned) > 0:
            mse = mean_squared_error(test_actual_aligned, test_pred_aligned)
            mae = mean_absolute_error(test_actual_aligned, test_pred_aligned)
            rmse = np.sqrt(mse)
            r2 = r2_score(test_actual_aligned, test_pred_aligned)
            
            print(f"\nTest Set Performance:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
            
            # Generate forecast
            forecast_horizon = config["evaluation"].get("forecast_horizon", len(test_df))
            forecast = forecast_multivariate(
                model,
                train_df,
                target_col=target_col,
                predictor_cols=[predictor_col],
                lag=lag,
                forecast_horizon=forecast_horizon,
            )
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax.plot(train_df.index, train_df[target_col], "k-", label=f"{target_col} (Train)", alpha=0.7)
            ax.plot(test_df.index, test_df[target_col], "g-", label=f"{target_col} (Test)", alpha=0.7)
            
            # Plot predictions
            if len(train_pred) > 0:
                ax.plot(train_pred.index, train_pred.values, "b--", label="Train Predictions", alpha=0.7)
            if len(test_pred_aligned) > 0:
                ax.plot(test_pred_aligned.index, test_pred_aligned.values, "r--", label="Test Predictions", alpha=0.7)
            
            # Plot forecast
            ax.plot(forecast.index, forecast.values, "m--", label="Forecast", linewidth=2)
            
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.set_title(f"Granger Causality Forecast: {target_col} using {predictor_col}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = output_dir / config["output"].get("plot_file", "granger_forecast.png")
            save_plot(fig, plot_path, dpi=config["output"].get("dpi", 300))
            print(f"\nPlot saved to: {plot_path}")
            
            # Save results
            results_df = pd.DataFrame({
                "date": forecast.index,
                "forecast": forecast.values,
            })
            csv_path = output_dir / config["output"].get("forecast_file", "granger_forecast.csv")
            results_df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"Forecast saved to: {csv_path}")
        else:
            print("\nWarning: Insufficient test data for evaluation")
    else:
        print(f"\nNo Granger causality detected. Cannot use {predictor_col} as leading indicator.")
        print("Consider using univariate forecasting methods instead.")
    
    # Save causality results
    results_summary = pd.DataFrame({
        "test": [f"{predictor_col} → {target_col}"],
        "min_p_value": [gc_results["min_p_value"]],
        "best_lag": [gc_results["best_lag"]],
        "is_causal": [gc_results["is_causal"]],
        "correlation": [correlation],
    })
    
    summary_path = output_dir / config["output"].get("summary_file", "granger_summary.csv")
    results_summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Causality results saved to: {summary_path}")


if __name__ == "__main__":
    main()

