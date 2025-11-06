#!/usr/bin/env python3
"""
PyCaret: Low-Code Time Series Forecasting
Automated time series forecasting with minimal code.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from pycaret.time_series import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot
from utils.ts_utils import load_ts_data

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_pycaret(data, config):
    """Setup PyCaret time series environment."""
    return setup(
        data=data,
        fh=config['model']['forecast_horizon'],
        session_id=config['model'].get('session_id', 42),
        verbose=False
    )


def create_and_compare_models(config):
    """Create and compare multiple models."""
    best_model = compare_models(
        include=config['model'].get('models', ['arima', 'exp_smooth', 'theta']),
        sort=config['model'].get('sort_metric', 'MAE'),
        verbose=False
    )
    
    return best_model


def finalize_model(model):
    """Finalize the best model."""
    return finalize_model(model)


def create_visualizations(model, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    plot_model(model, plot='forecast', save=True)
    
    fig, ax = setup_figure(config)
    
    plot_model(model, plot='forecast', display=False, save=False)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'pycaret_forecast.png', config)
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
    
    setup_pycaret(data, config)
    best_model = create_and_compare_models(config)
    final_model = finalize_model(best_model)
    create_visualizations(final_model, config)
    
    print("✓ PyCaret time series forecasting complete")
    print(f"Best model: {type(final_model).__name__}")


if __name__ == "__main__":
    main()

