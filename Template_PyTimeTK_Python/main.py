#!/usr/bin/env python3
"""
PyTimeTK for Time Series Analysis
Time series toolkit for feature engineering, visualization, and analysis.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
import pytimetk as tk

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_features(df, config):
    """Create time series features using pytimetk."""
    result_df = df.copy()
    
    if config['features']['rolling_windows']:
        for window in config['features']['rolling_windows']:
            result_df = tk.augment_rolling_apply(
                result_df,
                date_column=config['data']['date_col'],
                window=window,
                window_func=[
                    (f"rolling_mean_{window}", lambda x: x[config['data']['value_col']].mean()),
                    (f"rolling_std_{window}", lambda x: x[config['data']['value_col']].std()),
                ],
                center=config['features'].get('center', False),
                threads=config['features'].get('threads', 1)
            )
    
    if config['features']['fourier_terms']:
        result_df = tk.augment_fourier(
            result_df,
            date_column=config['data']['date_col'],
            periods=config['features'].get('fourier_periods', None),
            K=config['features'].get('fourier_K', 1)
        )
    
    if config['features']['lags']:
        result_df = tk.augment_lags(
            result_df,
            date_column=config['data']['date_col'],
            value_column=config['data']['value_col'],
            lags=config['features']['lags']
        )
    
    return result_df


def create_visualizations(df, features_df, config):
    """Generate visualizations using pytimetk."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config['plotting']['engine'] == 'plotly':
        fig = tk.plot_timeseries(
            df,
            date_column=config['data']['date_col'],
            value_column=config['data']['value_col'],
            facet_ncol=1,
            x_axis_date_labels=config['plotting'].get('date_format', "%Y-%m"),
            engine='plotly',
            title=config['plot_titles']['time_series']
        )
        output_path = output_dir / "pytimetk_timeseries.html"
        fig.write_html(str(output_path))
        print(f"Plot saved to {output_path}")
    else:
        fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
        apply_plot_style(ax, {'plotting': config['plotting']})
        
        ax.plot(df[config['data']['date_col']], df[config['data']['value_col']].values,
                'k-', linewidth=config['plotting']['linewidth'],
                alpha=config['plotting']['alpha'])
        ax.set_title(config['plot_titles']['time_series'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        
        output_path = output_dir / "pytimetk_timeseries.png"
        save_plot(fig, output_path)
        plt.show()
    
    if config['features']['rolling_windows']:
        fig, axes = plt.subplots(len(config['features']['rolling_windows']), 1,
                                figsize=(15, 5 * len(config['features']['rolling_windows'])))
        if len(config['features']['rolling_windows']) == 1:
            axes = [axes]
        
        for i, window in enumerate(config['features']['rolling_windows']):
            apply_plot_style(axes[i], {'plotting': config['plotting']})
            
            axes[i].plot(features_df[config['data']['date_col']],
                        features_df[f'rolling_mean_{window}'].values,
                        'b-', linewidth=config['plotting']['linewidth'],
                        alpha=config['plotting']['alpha'], label=f'Rolling Mean ({window})')
            axes[i].fill_between(
                features_df[config['data']['date_col']],
                features_df[f'rolling_mean_{window}'].values - features_df[f'rolling_std_{window}'].values,
                features_df[f'rolling_mean_{window}'].values + features_df[f'rolling_std_{window}'].values,
                alpha=0.2, color='blue', label=f'±1 Std ({window})'
            )
            axes[i].set_title(f'Rolling Statistics (Window={window})')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Value')
            apply_legend(axes[i], config['plotting']['legend'])
        
        plt.tight_layout()
        output_path = output_dir / "pytimetk_features.png"
        save_plot(fig, output_path)
        plt.show()


def main():
    """Main execution function."""
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col=config['data']['date_col'])
    
    if config['data']['filter_by_time']:
        start_date = config['data'].get('start_date', None)
        end_date = config['data'].get('end_date', None)
        if start_date or end_date:
            df = tk.filter_by_time(
                df,
                date_column=config['data']['date_col'],
                start_date=start_date,
                end_date=end_date
            )
    
    print("\nPyTimeTK Time Series Analysis:")
    print("=" * 70)
    print(f"Data points: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    features_df = create_features(df.reset_index(), config)
    
    print(f"\nFeatures created: {len(features_df.columns) - len(df.columns)}")
    print(f"Total columns: {len(features_df.columns)}")
    
    feature_cols = [col for col in features_df.columns if col not in [config['data']['date_col'], config['data']['value_col']]]
    print(f"\nFeature columns: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
    
    create_visualizations(df.reset_index(), features_df, config)
    
    if config['output']['save_features']:
        output_path = Path(__file__).parent / config['output']['output_dir'] / "features.csv"
        features_df.to_csv(output_path, index=False)
        print(f"\nFeatures saved to {output_path}")
    
    print("✓ PyTimeTK analysis complete")


if __name__ == "__main__":
    main()

