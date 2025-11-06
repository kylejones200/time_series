#!/usr/bin/env python3
"""
Confidence Intervals for Time Series Forecasts
Bootstrap and parametric confidence intervals for time series predictions.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index, split_ts

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def bootstrap_ci(data, n_bootstrap, alpha, forecast_steps, arima_order):
    """Generate bootstrap confidence intervals."""
    forecasts = []
    
    for _ in range(n_bootstrap):
        boot_data = resample(data, replace=True, n_samples=len(data))
        model = ARIMA(boot_data, order=arima_order).fit()
        forecast = model.forecast(steps=forecast_steps)
        forecasts.append(forecast)
    
    forecasts = np.array(forecasts)
    lower = np.percentile(forecasts, (alpha / 2) * 100, axis=0)
    upper = np.percentile(forecasts, (1 - alpha / 2) * 100, axis=0)
    mean = np.mean(forecasts, axis=0)
    
    return mean, lower, upper


def parametric_ci(data, forecast_steps, alpha, arima_order):
    """Generate parametric confidence intervals."""
    model = ARIMA(data, order=arima_order).fit()
    forecast = model.get_forecast(steps=forecast_steps)
    
    mean = forecast.predicted_mean
    ci = forecast.conf_int(alpha=alpha)
    lower = ci.iloc[:, 0].values
    upper = ci.iloc[:, 1].values
    
    return mean, lower, upper


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col=config['data']['date_col'])
    
    train_df, test_df = split_ts(df, test_size=config['data']['test_size'])
    train_data = train_df[config['data']['value_col']].values
    
    forecast_steps = len(test_df)
    alpha = config['model']['alpha']
    
    arima_order = tuple(config['model']['arima_order'])
    
    if config['model']['method'] == 'bootstrap':
        mean, lower, upper = bootstrap_ci(
            train_data, config['model']['n_bootstrap'], alpha, forecast_steps, arima_order
        )
    else:
        mean, lower, upper = parametric_ci(train_data, forecast_steps, alpha, arima_order)
    
    forecast_dates = test_df.index
    ci_level = (1 - alpha) * 100
    
    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, {'plotting': config['plotting']})
    
    ax.plot(train_df.index, train_df[config['data']['value_col']].values,
            'k-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Historical')
    ax.plot(test_df.index, test_df[config['data']['value_col']].values,
            'g-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Actual')
    ax.plot(forecast_dates, mean,
            'r--', linewidth=config['plotting']['linewidth'],
            label='Forecast')
    ax.fill_between(forecast_dates, lower, upper,
                     color='r', alpha=0.2, label=f'{ci_level:.0f}% CI')
    
    ax.set_title(config['plot_titles']['confidence_intervals'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config['plotting']['legend'])
    
    output_path = Path(__file__).parent / "outputs" / "confidence_intervals.png"
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()

