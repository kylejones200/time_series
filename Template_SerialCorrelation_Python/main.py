#!/usr/bin/env python3
"""
Serial Correlation Analysis for Time Series
Tests and corrections for serial correlation in time series regression models.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
from statsmodels.regression.linear_model import GLS, GLSAR
from statsmodels.graphics.tsaplots import plot_acf

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index, create_lags

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col=config['data']['date_col'])
    
    y_col = config['model']['y_col']
    x_cols = config['model']['x_cols']
    
    if config['model'].get('create_lags', False):
        for lag in range(1, config['model'].get('max_lags', 2) + 1):
            df = create_lags(df, x_cols[0], [lag])
            x_cols.append(f'{x_cols[0]}_lag{lag}')
    
    df = df[[y_col] + x_cols].dropna()
    
    y = df[y_col]
    X = df[x_cols]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    print("\nOLS Regression Results:")
    print("=" * 70)
    print(model.summary())
    
    residuals = model.resid
    
    bg_test = acorr_breusch_godfrey(model, nlags=config['model']['test_lags'])
    print(f"\nBreusch-Godfrey Test p-value: {bg_test[1]:.4f}")
    if bg_test[1] < 0.05:
        print("→ Serial correlation detected (p < 0.05)")
    else:
        print("→ No significant serial correlation (p >= 0.05)")
    
    lb_test = acorr_ljungbox(residuals, lags=config['model']['test_lags'], return_df=True)
    print(f"\nLjung-Box Test:")
    print(lb_test)
    
    if config['model'].get('apply_corrections', False):
        print("\n" + "=" * 70)
        print("Corrected Models:")
        print("=" * 70)
        
        gls_model = GLS(y, X).fit()
        print("\nGLS Model:")
        print(gls_model.summary())
        
        cochrane_orcutt = GLSAR(y, X, rho=1).iterative_fit()
        print("\nCochrane-Orcutt Model:")
        print(cochrane_orcutt.summary())
        
        model_robust = model.get_robustcov_results(cov_type="HAC", maxlags=config['model']['test_lags'])
        print("\nRobust Standard Errors (HAC):")
        print(model_robust.summary())
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    for ax in axes:
        apply_plot_style(ax, {'plotting': config['plotting']})
    
    axes[0].plot(df.index, residuals,
                 'k-', linewidth=config['plotting']['linewidth'],
                 alpha=config['plotting']['alpha'])
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0].set_title('Residuals Over Time')
    axes[0].set_ylabel('Residual')
    
    plot_acf(residuals, lags=config['model']['test_lags'], ax=axes[1], alpha=0.05)
    axes[1].set_title('Autocorrelation Function of Residuals')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Autocorrelation')
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "outputs" / "serial_correlation.png"
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()

