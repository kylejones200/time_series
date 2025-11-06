#!/usr/bin/env python3
"""
Bollinger Bands for Time Series Analysis
Technical indicator using moving averages and standard deviations.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def calculate_bollinger_bands(data, window, num_std):
    """Calculate Bollinger Bands."""
    df = data.copy()
    df['MA'] = df.rolling(window=window).mean()
    df['std'] = df.rolling(window=window).std()
    df['upper'] = df['MA'] + (df['std'] * num_std)
    df['lower'] = df['MA'] - (df['std'] * num_std)
    return df


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col=config['data']['date_col'])
    
    df_bb = calculate_bollinger_bands(
        df[config['data']['value_col']],
        config['model']['window'],
        config['model']['num_std']
    )
    
    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, {'plotting': config['plotting']})
    
    ax.plot(df.index, df[config['data']['value_col']].values,
            'k-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Price')
    ax.plot(df_bb.index, df_bb['MA'].values,
            'b-', linewidth=config['plotting']['linewidth'],
            label='Moving Average')
    ax.fill_between(df_bb.index, df_bb['lower'].values, df_bb['upper'].values,
                     color='b', alpha=0.2, label='Bollinger Bands')
    
    ax.set_title(config['plot_titles']['bollinger_bands'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config['plotting']['legend'])
    
    output_path = Path(__file__).parent / "outputs" / "bollinger_bands.png"
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()

