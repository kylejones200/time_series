#!/usr/bin/env python3
"""
N-BEATS: Neural Basis Expansion Analysis
Deep learning time series forecasting with interpretable basis expansion.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from darts.models import NBEATSModel
from darts import TimeSeries

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot
from utils.ts_utils import load_ts_data, split_ts

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config):
    """Create N-BEATS model."""
    return NBEATSModel(
        input_chunk_length=config['model'].get('input_chunk_length', 24),
        output_chunk_length=config['model'].get('output_chunk_length', 12),
        num_stacks=config['model'].get('num_stacks', 30),
        num_blocks=config['model'].get('num_blocks', 1),
        num_layers=config['model'].get('num_layers', 4),
        layer_widths=config['model'].get('layer_widths', 512),
        n_epochs=config['model'].get('n_epochs', 100),
    )


def fit_and_predict(model, train_data, config):
    """Fit model and generate predictions."""
    model.fit(train_data, verbose=False)
    
    forecast_horizon = config['model']['forecast_horizon']
    predictions = model.predict(forecast_horizon)
    
    return predictions


def create_visualizations(train_data, predictions, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = setup_figure(config)
    
    train_df = train_data.pd_dataframe()
    pred_df = predictions.pd_dataframe()
    
    ax.plot(train_df.index, train_df.values,
            c=config['plotting']['style']['colors']['primary'],
            linewidth=config['plotting']['style']['linewidth'],
            alpha=config['plotting']['style']['alpha'],
            label='Historical')
    
    ax.plot(pred_df.index, pred_df.values,
            c=config['plotting']['style']['colors']['secondary'],
            linewidth=config['plotting']['style']['linewidth'],
            label='Forecast')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'nbeats_forecast.png', config)
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
    train, _ = split_ts(ts.pd_series(), train_size=config['model'].get('train_size', 0.8))
    train_ts = TimeSeries.from_series(train)
    
    model = create_model(config)
    predictions = fit_and_predict(model, train_ts, config)
    create_visualizations(train_ts, predictions, config)
    
    print("✓ N-BEATS forecasting complete")


if __name__ == "__main__":
    main()

