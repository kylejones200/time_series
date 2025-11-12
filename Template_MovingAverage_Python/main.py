#!/usr/bin/env python3
"""
Moving Average: Simple Forecasting
Simple moving average and exponential moving average forecasting.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import util


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

ts_utils = repo_import("utils.ts_utils")
load_ts_data = ts_utils.load_ts_data
split_ts = ts_utils.split_ts
detect_frequency = ts_utils.detect_frequency


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def simple_moving_average(data, window):
    """Calculate simple moving average."""
    return data.rolling(window=window).mean()


def exponential_moving_average(data, alpha):
    """Calculate exponential moving average."""
    return data.ewm(alpha=alpha, adjust=False).mean()


def weighted_moving_average(data, window, weights=None):
    """Calculate weighted moving average."""
    weights = weights or np.ones(window) / window
    return data.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)


def create_forecast(data, config):
    """Create moving average forecast."""
    method = config['model']['method']
    window = config['model']['window']
    
    method_map = {
        'SMA': lambda: simple_moving_average(data, window),
        'EMA': lambda: exponential_moving_average(data, config['model'].get('alpha', 0.3)),
        'WMA': lambda: weighted_moving_average(data, window, config['model'].get('weights')),
    }
    
    ma = method_map.get(method, method_map['SMA'])()
    
    forecast_horizon = config['model']['forecast_horizon']
    last_value = ma.iloc[-1]
    
    forecast_index = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=detect_frequency(data)
    )
    
    forecast = pd.Series([last_value] * forecast_horizon, index=forecast_index)
    
    return ma, forecast


def create_visualizations(data, ma, forecast, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = setup_figure(config)
    
    ax.plot(data.index, data.values,
            c=config['plotting']['style']['colors']['primary'],
            linewidth=config['plotting']['style']['linewidth'],
            alpha=config['plotting']['style']['alpha'],
            label='Historical')
    
    ax.plot(ma.index, ma.values,
            c=config['plotting']['style']['colors']['accent'],
            linewidth=config['plotting']['style']['linewidth'],
            alpha=config['plotting']['style']['alpha'],
            label=f"{config['model']['method']}")
    
    ax.plot(forecast.index, forecast.values,
            c=config['plotting']['style']['colors']['secondary'],
            linewidth=config['plotting']['style']['linewidth'],
            label='Forecast')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'moving_average_forecast.png', config)
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
    
    ma, forecast = create_forecast(data, config)
    create_visualizations(data, ma, forecast, config)
    
    print("✓ Moving Average forecasting complete")


if __name__ == "__main__":
    main()

