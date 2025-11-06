#!/usr/bin/env python3
"""
ARAR: Autoregressive Autoregressive
Time series forecasting using reduced lag sets.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot
from utils.ts_utils import load_ts_data, split_ts, detect_frequency

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def select_lags(data, config):
    """Select reduced lag set based on autocorrelation."""
    acf_vals = acf(data, nlags=config['model'].get('max_lag', 20))
    
    lag_method = config['model'].get('lag_method', 'powers_of_2')
    
    method_map = {
        'powers_of_2': lambda: [2**i for i in range(int(np.log2(len(data)))) if 2**i <= config['model'].get('max_lag', 20)],
        'custom': lambda: config['model'].get('lags', [1, 2, 4, 8]),
        'auto': lambda: [i for i in range(1, min(10, len(data))) if abs(acf_vals[i]) > 0.1],
    }
    
    return method_map.get(lag_method, method_map['powers_of_2'])()


def create_arar_model(data, lags, config):
    """Create ARAR model with selected lags."""
    return AutoReg(data, lags=lags, old_names=False).fit()


def generate_forecast(model, data, forecast_horizon, config):
    """Generate forecasts and reverse differencing if needed."""
    forecasts = model.predict(start=len(data), end=len(data) + forecast_horizon - 1)
    
    if config['model'].get('differenced', False):
        forecasts = np.cumsum(forecasts) + data.iloc[-1] if hasattr(data, 'iloc') else np.cumsum(forecasts) + data[-1]
    
    return forecasts


def create_visualizations(data, train_data, forecasts, config):
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
        periods=len(forecasts),
        freq=detect_frequency(train_data)
    )
    
    ax.plot(forecast_index, forecasts,
            c=config['plotting']['style']['colors']['secondary'],
            linewidth=config['plotting']['style']['linewidth'],
            label='Forecast')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'arar_forecast.png', config)
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
    
    if config['model'].get('differenced', False):
        train_diff = np.diff(train_data.values)
        train_diff_series = pd.Series(train_diff, index=train_data.index[1:])
    else:
        train_diff_series = train_data
    
    lags = select_lags(train_diff_series, config)
    model = create_arar_model(train_diff_series, lags, config)
    forecasts = generate_forecast(model, train_diff_series, config['model']['forecast_horizon'], config)
    create_visualizations(data, train_data, forecasts, config)
    
    print("✓ ARAR forecasting complete")


if __name__ == "__main__":
    main()

