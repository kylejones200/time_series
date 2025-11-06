#!/usr/bin/env python3
"""
Darts: Time Series Forecasting Library
Unified interface for multiple forecasting models.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from darts import TimeSeries
from darts.models import (
    ExponentialSmoothing, ARIMA, Prophet, NBEATS, 
    RandomForest, XGBModel, LightGBMModel
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot
from utils.ts_utils import load_ts_data, split_ts

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config):
    """Create Darts model based on config."""
    model_map = {
        'ExponentialSmoothing': ExponentialSmoothing,
        'ARIMA': ARIMA,
        'Prophet': Prophet,
        'NBEATS': NBEATS,
        'RandomForest': RandomForest,
        'XGBModel': XGBModel,
        'LightGBMModel': LightGBMModel,
    }
    
    model_class = model_map[config['model']['type']]
    model_params = config['model'].get('params', {})
    
    return model_class(**model_params)


def fit_and_predict(model, train_data, config):
    """Fit model and generate predictions."""
    model.fit(train_data)
    
    forecast_horizon = config['model']['forecast_horizon']
    predictions = model.predict(forecast_horizon)
    
    return predictions


def create_visualizations(train_data, test_data, predictions, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = setup_figure(config)
    
    train_df = train_data.pd_dataframe()
    test_df = test_data.pd_dataframe() if test_data else None
    pred_df = predictions.pd_dataframe()
    
    ax.plot(train_df.index, train_df.values,
            c=config['plotting']['style']['colors']['primary'],
            linewidth=config['plotting']['style']['linewidth'],
            alpha=config['plotting']['style']['alpha'],
            label='Train')
    
    [ax.plot(test_df.index, test_df.values,
             c=config['plotting']['style']['colors']['accent'],
             linewidth=config['plotting']['style']['linewidth'],
             alpha=config['plotting']['style']['alpha'],
             label='Test')
     for _ in [None] if test_data is not None]
    
    ax.plot(pred_df.index, pred_df.values,
            c=config['plotting']['style']['colors']['secondary'],
            linewidth=config['plotting']['style']['linewidth'],
            label='Forecast')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'darts_forecast.png', config)
     for _ in [None] if config['output']['save_plots']]
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    data = load_ts_data(
        Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    
    ts = TimeSeries.from_series(data)
    train, test = split_ts(ts.pd_series(), train_size=config['model'].get('train_size', 0.8))
    
    train_ts = TimeSeries.from_series(train)
    test_ts = TimeSeries.from_series(test) if test is not None else None
    
    model = create_model(config)
    predictions = fit_and_predict(model, train_ts, config)
    create_visualizations(train_ts, test_ts, predictions, config)
    
    print("✓ Darts forecasting complete")


if __name__ == "__main__":
    main()

