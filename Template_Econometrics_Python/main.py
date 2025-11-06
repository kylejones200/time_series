#!/usr/bin/env python3
"""
Econometric Time Series Analysis
Causal inference, policy evaluation, and economic modeling methods.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index, split_ts

warnings.filterwarnings('ignore')


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
                p_value = gc_result[lag][0]['ssr_ftest'][1]
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
    
    df['time_from_cutoff'] = (df[date_col] - cutoff).dt.days
    df['treatment'] = (df['time_from_cutoff'] >= 0).astype(int)
    df['interaction'] = df['time_from_cutoff'] * df['treatment']
    
    model = ols(f'{value_col} ~ time_from_cutoff * treatment', data=df).fit()
    
    print(model.summary())
    
    treatment_effect = model.params['treatment']
    print(f"\nTreatment Effect: {treatment_effect:.4f}")
    
    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, {'plotting': config['plotting']})
    
    pre_treatment = df[df['treatment'] == 0]
    post_treatment = df[df['treatment'] == 1]
    
    ax.scatter(pre_treatment['time_from_cutoff'], pre_treatment[value_col],
               alpha=config['plotting']['alpha'], s=config['plotting']['markersize'] * 10,
               label='Pre-treatment', c='b')
    ax.scatter(post_treatment['time_from_cutoff'], post_treatment[value_col],
               alpha=config['plotting']['alpha'], s=config['plotting']['markersize'] * 10,
               label='Post-treatment', c='r')
    ax.axvline(x=0, color='k', linestyle='--', linewidth=2, label='Cutoff')
    
    ax.set_title(config['plot_titles']['rdd_analysis'])
    ax.set_xlabel('Time from Cutoff (days)')
    ax.set_ylabel('Value')
    apply_legend(ax, config['plotting']['legend'])
    
    output_path = Path(__file__).parent / "outputs" / "rdd_analysis.png"
    save_plot(fig, output_path)
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
    lag_order = model.select_order(maxlags=config['model']['var_max_lags'])
    optimal_lag = lag_order.aic
    
    print(f"\nOptimal lag order (AIC): {optimal_lag}")
    
    fitted_model = model.fit(optimal_lag)
    print(fitted_model.summary())
    
    return fitted_model, optimal_lag


def main():
    config = load_config()
    
    data_path = Path(__file__).parent.parent / 'data' / config['data']['input_file']
    df = pd.read_csv(data_path)
    df[config['data']['date_col']] = pd.to_datetime(df[config['data']['date_col']])
    df = df.set_index(config['data']['date_col']).sort_index()
    
    method = config['model']['method']
    
    method_map = {
        'granger': lambda: granger_causality_test(
            df, config['model']['x_col'], config['model']['y_col'],
            config['model']['max_lag'], config
        ),
        'rdd': lambda: regression_discontinuity(
            df, config['data']['date_col'], config['data']['value_col'],
            config['model']['cutoff_date'], config
        ),
        'ols': lambda: ols_regression(
            df, config['model']['y_col'], config['model']['x_cols'], config
        ),
        'var': lambda: var_analysis(
            df, config['data']['value_cols'], config
        ),
    }
    
    result = method_map.get(method, lambda: print(f"Unknown method: {method}"))()
    print(f"\n✓ Econometric analysis ({method}) complete")


if __name__ == "__main__":
    main()

