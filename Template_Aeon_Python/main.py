#!/usr/bin/env python3
"""
Aeon: Time Series Analysis Toolkit
Comprehensive toolkit for time series analysis, classification, and forecasting.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from aeon.datasets import load_airline
from aeon.forecasting.trend import PolynomialTrendForecaster
from aeon.forecasting.arima import ARIMA
from aeon.classification import TimeSeriesForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(config):
    """Load time series data."""
    data_path = Path(__file__).parent.parent / 'data' / Path(config['data']['input_file']).name
    
    data_loader_map = {
        'airline': load_airline,
        'csv': lambda: pd.read_csv(data_path)
    }
    
    loader = data_loader_map.get(config['data']['source'], lambda: pd.read_csv(data_path))
    data = loader()
    
    [data.__setitem__(config['data']['date_col'], pd.to_datetime(data[config['data']['date_col']]))
     for _ in [None] if config['data'].get('date_col')]
    
    return data


def create_forecaster(config):
    """Create Aeon forecaster based on config."""
    forecaster_map = {
        'PolynomialTrend': PolynomialTrendForecaster,
        'ARIMA': ARIMA,
    }
    
    forecaster_class = forecaster_map[config['model']['type']]
    forecaster_params = config['model'].get('params', {})
    
    return forecaster_class(**forecaster_params)


def fit_and_predict(forecaster, data, config):
    """Fit forecaster and generate predictions."""
    forecaster.fit(data)
    fh = np.arange(1, config['model']['forecast_horizon'] + 1)
    predictions = forecaster.predict(fh=fh)
    return predictions


def create_visualizations(data, predictions, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = setup_figure(config)
    
    ax.plot(data.index, data.values,
            c=config['plotting']['style']['colors']['primary'],
            linewidth=config['plotting']['style']['linewidth'],
            alpha=config['plotting']['style']['alpha'],
            label='Historical')
    
    ax.plot(predictions.index, predictions.values,
            c=config['plotting']['style']['colors']['secondary'],
            linewidth=config['plotting']['style']['linewidth'],
            label='Forecast')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'aeon_forecast.png', config)
     for _ in [None] if config['output']['save_plots']]
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    data = load_data(config)
    forecaster = create_forecaster(config)
    predictions = fit_and_predict(forecaster, data, config)
    create_visualizations(data, predictions, config)
    
    print("✓ Aeon forecasting complete")


if __name__ == "__main__":
    main()

