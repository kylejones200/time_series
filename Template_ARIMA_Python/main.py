#!/usr/bin/env python3
"""
ARIMA: Autoregressive Integrated Moving Average
Classical time series forecasting model.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot
from utils.ts_utils import load_ts_data, split_ts, detect_frequency

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_stationarity(data):
    """Check if time series is stationary."""
    result = adfuller(data.values)
    return result[1] < 0.05


def auto_arima(data, config):
    """Auto-select ARIMA order using pmdarima."""
    return pm.auto_arima(
        data,
        start_p=config['model'].get('start_p', 0),
        start_q=config['model'].get('start_q', 0),
        max_p=config['model'].get('max_p', 5),
        max_q=config['model'].get('max_q', 5),
        seasonal=config['model'].get('seasonal', False),
        stepwise=config['model'].get('stepwise', True),
        suppress_warnings=True,
        error_action='ignore'
    )


def create_arima_model(data, config):
    """Create ARIMA model."""
    model_type = config['model']['type']
    
    model_map = {
        'auto': lambda: auto_arima(data, config),
        'manual': lambda: ARIMA(data, order=tuple(config['model']['order'])),
    }
    
    return model_map.get(model_type, model_map['auto'])()


def fit_and_predict(model, config):
    """Fit model and generate predictions."""
    fitted_model = model.fit() if hasattr(model, 'fit') else model
    
    forecast_horizon = config['model']['forecast_horizon']
    predictions = fitted_model.predict(n_periods=forecast_horizon)
    
    return fitted_model, predictions


def create_visualizations(data, train_data, predictions, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = setup_figure(config)
    
    ax.plot(train_data.index, train_data.values,
            c=config['plotting']['style']['colors']['primary'],
            linewidth=config['plotting']['style']['linewidth'],
            alpha=config['plotting']['style']['alpha'],
            label='Historical')
    
    forecast_index = pd.date_range(
        start=train_data.index[-1] + pd.Timedelta(days=1),
        periods=len(predictions),
        freq=detect_frequency(train_data)
    )
    
    ax.plot(forecast_index, predictions,
            c=config['plotting']['style']['colors']['secondary'],
            linewidth=config['plotting']['style']['linewidth'],
            label='Forecast')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'arima_forecast.png', config)
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
    
    train_size = config['model'].get('train_size', 0.8)
    train_data, _ = split_ts(data, train_size=train_size)
    
    model = create_arima_model(train_data, config)
    fitted_model, predictions = fit_and_predict(model, config)
    create_visualizations(data, train_data, predictions, config)
    
    print(f"✓ ARIMA forecasting complete")
    print(f"Model order: {fitted_model.order if hasattr(fitted_model, 'order') else 'Auto-selected'}")


if __name__ == "__main__":
    main()

