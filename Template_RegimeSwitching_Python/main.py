#!/usr/bin/env python3
"""
Regime Switching Models for Time Series
Markov switching models for time series with structural breaks.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

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
    
    data = df[config['data']['value_col']].values
    
    model = MarkovRegression(
        data,
        k_regimes=config['model']['k_regimes'],
        trend=config['model']['trend'],
        switching_variance=config['model']['switching_variance']
    )
    
    result = model.fit()
    print("\nMarkov Switching Model Results:")
    print("=" * 70)
    print(result.summary())
    
    print("\nTransition Matrix:")
    print(result.regime_transition)
    
    smoothed_probs = result.smoothed_marginal_probabilities
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    for ax in axes:
        apply_plot_style(ax, {'plotting': config['plotting']})
    
    axes[0].plot(df.index, data,
                 'k-', linewidth=config['plotting']['linewidth'],
                 alpha=config['plotting']['alpha'], label='Time Series')
    axes[0].set_title(config['plot_titles']['regime_switching'])
    axes[0].set_ylabel('Value')
    apply_legend(axes[0], config['plotting']['legend'])
    
    for regime in range(config['model']['k_regimes']):
        axes[1].plot(df.index, smoothed_probs[:, regime],
                     linewidth=config['plotting']['linewidth'],
                     alpha=config['plotting']['alpha'],
                     label=f'Regime {regime + 1} Probability')
    axes[1].set_title('Smoothed Regime Probabilities')
    axes[1].set_ylabel('Probability')
    axes[1].set_xlabel('Date')
    apply_legend(axes[1], config['plotting']['legend'])
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "outputs" / "regime_switching.png"
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()

