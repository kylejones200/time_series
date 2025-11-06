#!/usr/bin/env python3
"""
Time Series Decomposition
Decompose time series into trend, seasonal, and residual components.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col=config['data']['date_col'])
    
    data = df[config['data']['value_col']]
    
    decomposition = seasonal_decompose(
        data,
        model=config['model']['decomposition_model'],
        period=config['model']['period'],
        extrapolate_trend='freq'
    )
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    for ax in axes:
        apply_plot_style(ax, {'plotting': config['plotting']})
    
    axes[0].plot(data.index, data.values,
                 'k-', linewidth=config['plotting']['linewidth'],
                 alpha=config['plotting']['alpha'])
    axes[0].set_title('Original Time Series')
    axes[0].set_ylabel('Value')
    
    axes[1].plot(decomposition.trend.index, decomposition.trend.values,
                 'b-', linewidth=config['plotting']['linewidth'],
                 alpha=config['plotting']['alpha'])
    axes[1].set_title('Trend Component')
    axes[1].set_ylabel('Trend')
    
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values,
                 'g-', linewidth=config['plotting']['linewidth'],
                 alpha=config['plotting']['alpha'])
    axes[2].set_title('Seasonal Component')
    axes[2].set_ylabel('Seasonal')
    
    axes[3].plot(decomposition.resid.index, decomposition.resid.values,
                 'r-', linewidth=config['plotting']['linewidth'],
                 alpha=config['plotting']['alpha'])
    axes[3].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[3].set_title('Residual Component')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "outputs" / "time_series_decomposition.png"
    save_plot(fig, output_path)
    plt.show()
    
    print("\nDecomposition Statistics:")
    print("=" * 70)
    print(f"Trend range: [{decomposition.trend.min():.2f}, {decomposition.trend.max():.2f}]")
    print(f"Seasonal range: [{decomposition.seasonal.min():.2f}, {decomposition.seasonal.max():.2f}]")
    print(f"Residual std: {decomposition.resid.std():.4f}")


if __name__ == "__main__":
    main()

