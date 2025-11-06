#!/usr/bin/env python3
"""
Box-Jenkins Methodology for ARIMA Modeling
Systematic approach to ARIMA model identification, estimation, and diagnostics.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index, split_ts

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_differencing_order(timeseries, max_d=3):
    """Find optimal differencing order using ADF test."""
    current_series = timeseries.copy()
    for d in range(max_d + 1):
        result = adfuller(current_series.dropna(), autolag='AIC')
        if result[1] <= 0.05:
            return d, current_series
        current_series = current_series.diff()
    return max_d, current_series


def fit_arima_model(train_data, config):
    """Fit ARIMA model using auto_arima or manual parameters."""
    if config['model']['use_auto_arima']:
        model = auto_arima(
            train_data,
            start_p=config['model']['start_p'],
            start_q=config['model']['start_q'],
            max_p=config['model']['max_p'],
            max_q=config['model']['max_q'],
            d=config['model']['d'],
            seasonal=config['model']['seasonal'],
            stepwise=True,
            suppress_warnings=True,
            trace=False,
            error_action='ignore'
        )
    else:
        order = (config['model']['p'], config['model']['d'], config['model']['q'])
        model = ARIMA(train_data, order=order).fit()
    
    return model


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col='date')
    
    train_df, test_df = split_ts(df, test_size=config['data']['test_size'])
    train_data = train_df[config['data']['value_col']]
    
    d_order, differenced_data = find_differencing_order(train_data, max_d=config['model']['max_d'])
    print(f"\nOptimal differencing order (d): {d_order}")
    
    if config['model']['d'] is None:
        config['model']['d'] = d_order
    
    model = fit_arima_model(train_data, config)
    print(f"\nBest Model: ARIMA{model.order}")
    print(f"AIC: {model.aic():.2f}")
    
    residuals = model.resid()
    lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    print("\nLjung-Box Test for Residual Autocorrelation:")
    print(lb_test)
    
    forecast_result = model.predict(n_periods=len(test_df), return_conf_int=True, alpha=0.05)
    forecast = forecast_result[0]
    conf_int = forecast_result[1]
    
    mae = mean_absolute_error(test_df[config['data']['value_col']].values, forecast)
    rmse = np.sqrt(mean_squared_error(test_df[config['data']['value_col']].values, forecast))
    r2 = r2_score(test_df[config['data']['value_col']].values, forecast)
    
    print(f"\nModel Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(train_data.index, train_data.values, 'k-', linewidth=1.5)
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    axes[0, 0].set_title('Original Series')
    
    axes[0, 1].plot(differenced_data.index, differenced_data.values, 'r-', linewidth=1.5)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Differenced')
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)
    axes[0, 1].set_title('Differenced Series')
    
    plot_acf(differenced_data.dropna(), lags=40, ax=axes[1, 0])
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)
    axes[1, 0].set_title('ACF')
    
    plot_pacf(differenced_data.dropna(), lags=40, ax=axes[1, 1])
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
    axes[1, 1].set_title('PACF')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "box_jenkins_diagnostics.png"
    plt.savefig(output_path, dpi=config['plotting']['dpi'], bbox_inches='tight')
    plt.show()
    
    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, config)
    
    ax.plot(train_data.index[-100:], train_data.values[-100:],
            'k-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Historical')
    ax.plot(test_df.index, test_df[config['data']['value_col']].values,
            'g-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Actual (Test)')
    ax.plot(test_df.index, forecast,
            'r--', linewidth=config['plotting']['linewidth'],
            label='Forecast')
    ax.fill_between(test_df.index, conf_int[:, 0], conf_int[:, 1],
                     color='r', alpha=0.2, label='95% CI')
    
    ax.set_title(config['plot_titles']['box_jenkins_forecast'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config['plotting']['legend'])
    
    output_path = Path(__file__).parent / "outputs" / "box_jenkins_forecast.png"
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()

