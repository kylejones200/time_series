#!/usr/bin/env python3
"""
AutoGluon for Time Series Forecasting
Automated time series forecasting with AutoGluon's TimeSeriesPredictor.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from autogluon.timeseries import TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index, split_ts

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_autogluon_data(df, date_col, value_col, item_id_col=None, default_item_id='default'):
    """Prepare data for AutoGluon TimeSeriesPredictor."""
    df = df.copy()
    df = df.reset_index()
    
    if item_id_col is None:
        df['item_id'] = default_item_id
    else:
        df['item_id'] = df[item_id_col]
    
    df = df.rename(columns={date_col: 'timestamp', value_col: 'target'})
    df = df[['item_id', 'timestamp', 'target']]
    df = df.set_index(['item_id', 'timestamp'])
    
    return df


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col=config['data']['date_col'])
    
    train_df, test_df = split_ts(df, test_size=config['data']['test_size'])
    
    train_ag = prepare_autogluon_data(
        train_df, config['data']['date_col'], 
        config['data']['value_col'], 
        config['data'].get('item_id_col'),
        config['data'].get('item_id', 'default')
    )
    test_ag = prepare_autogluon_data(
        test_df, config['data']['date_col'],
        config['data']['value_col'],
        config['data'].get('item_id_col'),
        config['data'].get('item_id', 'default')
    )
    
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=config['model']['prediction_length'],
        freq=config['model'].get('freq', 'D'),
        path=str(Path(__file__).parent / config['model']['model_path']),
        eval_metric=config['model'].get('eval_metric', 'MAPE'),
    )
    
    if config['model'].get('presets'):
        predictor.fit(
            train_data=train_ag,
            presets=config['model']['presets'],
            hyperparameters=config['model'].get('hyperparameters', {})
        )
    else:
        predictor.fit(
            train_data=train_ag,
            hyperparameters=config['model'].get('hyperparameters', {})
        )
    
    forecast = predictor.predict(train_ag)
    
    test_values = test_ag['target'].values
    forecast_values = forecast['mean'].values[:len(test_values)]
    
    mae = mean_absolute_error(test_values, forecast_values)
    rmse = np.sqrt(mean_squared_error(test_values, forecast_values))
    r2 = r2_score(test_values, forecast_values)
    
    print(f"\nModel Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, {'plotting': config['plotting']})
    
    ax.plot(train_df.index, train_df[config['data']['value_col']].values,
            'k-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Historical')
    ax.plot(test_df.index, test_values,
            'g-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Actual (Test)')
    ax.plot(test_df.index[:len(forecast_values)], forecast_values,
            'r--', linewidth=config['plotting']['linewidth'],
            label='Forecast')
    
    ax.set_title(config['plot_titles']['autogluon_forecast'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config['plotting']['legend'])
    
    output_path = Path(__file__).parent / "outputs" / "autogluon_forecast.png"
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()

