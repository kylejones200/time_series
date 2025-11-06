#!/usr/bin/env python3
"""
Exponential Smoothing for Time Series Forecasting
Simple, Double, and Triple (Holt-Winters) exponential smoothing using statsmodels.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index, split_ts

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def fit_exponential_smoothing(train_data, config):
    """Fit exponential smoothing model based on type."""
    model_type = config['model']['type']
    
    model_map = {
        'simple': lambda: SimpleExpSmoothing(train_data).fit(
            smoothing_level=config['model']['smoothing_level'],
            optimized=config['model']['optimized']
        ),
        'double': lambda: ExponentialSmoothing(
            train_data, trend='add', seasonal=None
        ).fit(
            smoothing_level=config['model']['smoothing_level'],
            smoothing_trend=config['model']['smoothing_trend'],
            optimized=config['model']['optimized']
        ),
        'triple': lambda: ExponentialSmoothing(
            train_data,
            trend=config['model']['trend'],
            seasonal=config['model']['seasonal'],
            seasonal_periods=config['model']['seasonal_periods']
        ).fit(
            smoothing_level=config['model']['smoothing_level'],
            smoothing_trend=config['model']['smoothing_trend'],
            smoothing_seasonal=config['model']['smoothing_seasonal'],
            optimized=config['model']['optimized']
        ),
    }
    
    return model_map.get(model_type, model_map['simple'])()


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col='date')
    
    train_df, test_df = split_ts(df, test_size=config['data']['test_size'])
    
    model = fit_exponential_smoothing(train_df[config['data']['value_col']].values, config)
    
    train_fitted = model.fittedvalues
    forecast = model.forecast(steps=len(test_df))
    
    mae = mean_absolute_error(test_df[config['data']['value_col']].values, forecast)
    rmse = np.sqrt(mean_squared_error(test_df[config['data']['value_col']].values, forecast))
    r2 = r2_score(test_df[config['data']['value_col']].values, forecast)
    
    print(f"\nModel Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, config)
    
    ax.plot(train_df.index, train_df[config['data']['value_col']].values,
            'k-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Historical Data')
    ax.plot(train_df.index, train_fitted,
            'b--', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Fitted')
    ax.plot(test_df.index, test_df[config['data']['value_col']].values,
            'g-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Actual (Test)')
    ax.plot(test_df.index, forecast,
            'r-', linewidth=config['plotting']['linewidth'],
            label='Forecast')
    
    ax.set_title(config['plot_titles']['exponential_smoothing'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config['plotting']['legend'])
    
    output_path = Path(__file__).parent / "outputs" / "exponential_smoothing_forecast.png"
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()

