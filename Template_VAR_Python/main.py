#!/usr/bin/env python3
"""
Vector Autoregression (VAR) for Multivariate Time Series
VAR modeling for multiple interdependent time series using statsmodels.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import util
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj


plotting_utils = repo_import("utils.plotting_utils")
setup_figure = plotting_utils.setup_figure
apply_legend = plotting_utils.apply_legend
save_plot = plotting_utils.save_plot
apply_plot_style = plotting_utils.apply_plot_style

ts_utils = repo_import("utils.ts_utils")
load_ts_data = ts_utils.load_ts_data
ensure_datetime_index = ts_utils.ensure_datetime_index
split_ts = ts_utils.split_ts


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_stationarity(data, config):
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
    
    if differenced or config['model']['force_differencing']:
        stationary_data = data.diff().dropna()
        print("\nApplied first-order differencing")
    
    return stationary_data


def fit_var_model(train_data, config):
    """Fit VAR model with optimal lag selection."""
    model = VAR(train_data)
    lag_order = model.select_order(maxlags=config['model']['max_lags'])
    
    optimal_lag = lag_order.aic if config['model']['use_aic'] else lag_order.bic
    print(f"\nOptimal lag order: {optimal_lag}")
    
    fitted_model = model.fit(optimal_lag)
    print(f"\nVAR({optimal_lag}) Model Summary:")
    print(fitted_model.summary())
    
    return fitted_model, optimal_lag


def granger_causality_test(model, data, config):
    """Perform Granger causality tests."""
    if config['model']['granger_test']:
        print("\nGranger Causality Tests:")
        for col1 in data.columns:
            for col2 in data.columns:
                if col1 != col2:
                    print(f"\nTesting if {col2} Granger-causes {col1}:")
                    try:
                        gc_result = grangercausalitytests(
                            data[[col1, col2]],
                            maxlag=config['model']['granger_maxlag'],
                            verbose=False
                        )
                        p_values = [gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, config['model']['granger_maxlag'] + 1)]
                        min_p = min(p_values)
                        print(f"  Minimum p-value: {min_p:.4f}", end="")
                        if min_p < 0.05:
                            print(" → Significant Granger causality")
                        else:
                            print(" → No significant Granger causality")
                    except Exception as e:
                        print(f"  Error: {e}")


def main():
    config = load_config()
    
    data_path = Path(__file__).parent.parent / 'data' / config['data']['input_file']
    df = pd.read_csv(data_path)
    df[config['data']['date_col']] = pd.to_datetime(df[config['data']['date_col']])
    df = df.set_index(config['data']['date_col']).sort_index()
    df = df[config['data']['value_cols']]
    
    print("\nStationarity Tests:")
    stationary_df = check_stationarity(df[config['data']['value_cols']], config)
    
    train_df, test_df = split_ts(stationary_df, test_size=config['data']['test_size'])
    
    model, optimal_lag = fit_var_model(train_df, config)
    
    if config['model']['granger_test']:
        granger_causality_test(model, train_df, config)
    
    residuals = model.resid
    print("\nDurbin-Watson Test (should be close to 2):")
    for col in residuals.columns:
        dw_stat = durbin_watson(residuals[col])
        print(f"  {col}: {dw_stat:.2f}")
    
    forecast = model.forecast(train_df.values[-optimal_lag:], steps=len(test_df))
    forecast_df = pd.DataFrame(
        forecast,
        index=test_df.index,
        columns=test_df.columns
    )
    
    if config['model']['force_differencing'] or stationary_df is not df[config['data']['value_cols']]:
        forecast_actual = forecast_df.cumsum() + df[config['data']['value_cols']].iloc[-len(test_df)-1].values
        forecast_actual = pd.DataFrame(forecast_actual, index=test_df.index, columns=test_df.columns)
    else:
        forecast_actual = forecast_df
    
    metrics = {}
    for col in test_df.columns:
        if col in df.columns:
            actual = df[col].iloc[-len(test_df):].values
            pred = forecast_actual[col].values
            metrics[col] = {
                'MAE': mean_absolute_error(actual, pred),
                'RMSE': np.sqrt(mean_squared_error(actual, pred)),
                'R²': r2_score(actual, pred)
            }
    
    print("\nModel Evaluation:")
    for col, m in metrics.items():
        print(f"\n{col}:")
        [print(f"  {k}: {v:.4f}") for k, v in m.items()]
    
    n_vars = len(config['data']['value_cols'])
    fig, axes = plt.subplots(n_vars, 1, figsize=(15, 5 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]
    
    for i, col in enumerate(config['data']['value_cols']):
        axes[i].plot(df.index[-100:], df[col].values[-100:],
                     'k-', linewidth=2, alpha=0.7, label='Historical')
        if col in test_df.columns:
            axes[i].plot(test_df.index, df[col].iloc[-len(test_df):].values,
                         'g-', linewidth=2, alpha=0.7, label='Actual (Test)')
        if col in forecast_actual.columns:
            axes[i].plot(forecast_actual.index, forecast_actual[col].values,
                         'r--', linewidth=2, label='Forecast')
        
        axes[i].set_title(f'VAR Forecast: {col}')
        axes[i].set_ylabel('Value')
        axes[i].legend(loc='best')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "outputs" / "var_forecast.png"
    plt.savefig(output_path, dpi=config['plotting']['dpi'], bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

