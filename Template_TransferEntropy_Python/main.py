#!/usr/bin/env python3
"""
Transfer Entropy for Causal Inference
Information-theoretic causal inference using transfer entropy.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
import pyinform.transferentropy as te
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def discretize(series, bins):
    """Discretize continuous series into bins."""
    return pd.qcut(series.rank(method="first"), bins, labels=False).astype(int)


def compute_transfer_entropy(source, target, k, bins):
    """Compute transfer entropy between two series."""
    source_disc = discretize(source, bins).values.reshape(1, -1)
    target_disc = discretize(target, bins).values.reshape(1, -1)
    
    try:
        te_value = te.transfer_entropy(source_disc, target_disc, k=k)
        return te_value if not np.isnan(te_value) else 0.0
    except Exception:
        return 0.0


def compute_rolling_te(series1, series2, window, k, bins, config):
    """Compute rolling transfer entropy."""
    te_1_to_2 = []
    te_2_to_1 = []
    dates = []
    
    for i in tqdm(range(window, len(series1)), desc="Computing rolling TE"):
        sub1 = series1.iloc[i - window:i]
        sub2 = series2.iloc[i - window:i]
        
        te_12 = compute_transfer_entropy(sub1, sub2, k, bins)
        te_21 = compute_transfer_entropy(sub2, sub1, k, bins)
        
        te_1_to_2.append(te_12)
        te_2_to_1.append(te_21)
        dates.append(series1.index[i])
    
    return pd.DataFrame({
        f'{config["data"]["series1_name"]} → {config["data"]["series2_name"]}': te_1_to_2,
        f'{config["data"]["series2_name"]} → {config["data"]["series1_name"]}': te_2_to_1
    }, index=pd.to_datetime(dates))


def create_visualizations(series1, series2, te_df, static_te, config):
    """Generate visualizations for transfer entropy."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for ax in axes.flatten():
        apply_plot_style(ax, {'plotting': config['plotting']})
    
    axes[0, 0].plot(series1.index, series1.values,
                    'k-', linewidth=config['plotting']['linewidth'],
                    alpha=config['plotting']['alpha'], label=config['data']['series1_name'])
    axes[0, 0].plot(series2.index, series2.values,
                    'r-', linewidth=config['plotting']['linewidth'],
                    alpha=config['plotting']['alpha'], label=config['data']['series2_name'])
    axes[0, 0].set_title('Time Series')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value')
    apply_legend(axes[0, 0], config['plotting']['legend'])
    
    axes[0, 1].plot(te_df.index, te_df.iloc[:, 0],
                    'k-', linewidth=config['plotting']['linewidth'],
                    alpha=config['plotting']['alpha'], label=te_df.columns[0])
    axes[0, 1].plot(te_df.index, te_df.iloc[:, 1],
                    'r-', linewidth=config['plotting']['linewidth'],
                    alpha=config['plotting']['alpha'], label=te_df.columns[1])
    axes[0, 1].set_title('Rolling Transfer Entropy')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Transfer Entropy')
    apply_legend(axes[0, 1], config['plotting']['legend'])
    
    axes[1, 0].bar([0, 1], [static_te['te_1_to_2'], static_te['te_2_to_1']],
                   edgecolor='black', alpha=0.7, width=0.6)
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels([te_df.columns[0], te_df.columns[1]], rotation=45, ha='right')
    axes[1, 0].set_title('Static Transfer Entropy')
    axes[1, 0].set_ylabel('Transfer Entropy')
    
    axes[1, 1].scatter(te_df.iloc[:, 0], te_df.iloc[:, 1],
                      alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
    axes[1, 1].set_xlabel(te_df.columns[0])
    axes[1, 1].set_ylabel(te_df.columns[1])
    axes[1, 1].set_title('Bidirectional Transfer Entropy')
    
    plt.tight_layout()
    
    output_path = output_dir / "transfer_entropy_analysis.png"
    save_plot(fig, output_path)
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    
    df1 = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['series1_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['series1_col']
    )
    df1 = ensure_datetime_index(df1, time_col=config['data']['date_col'])
    
    df2 = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['series2_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['series2_col']
    )
    df2 = ensure_datetime_index(df2, time_col=config['data']['date_col'])
    
    series1 = df1[config['data']['series1_col']]
    series2 = df2[config['data']['series2_col']]
    
    if config['data']['difference']:
        series1 = series1.diff().dropna()
        series2 = series2.diff().dropna()
    
    common_index = series1.index.intersection(series2.index)
    series1 = series1.loc[common_index]
    series2 = series2.loc[common_index]
    
    static_te_1_to_2 = compute_transfer_entropy(
        series1, series2,
        config['model']['k'],
        config['model']['bins']
    )
    static_te_2_to_1 = compute_transfer_entropy(
        series2, series1,
        config['model']['k'],
        config['model']['bins']
    )
    
    print("\nTransfer Entropy Results:")
    print("=" * 70)
    print(f"{config['data']['series1_name']} → {config['data']['series2_name']}: {static_te_1_to_2:.4f}")
    print(f"{config['data']['series2_name']} → {config['data']['series1_name']}: {static_te_2_to_1:.4f}")
    
    if static_te_1_to_2 > static_te_2_to_1:
        print(f"\nStronger direction: {config['data']['series1_name']} → {config['data']['series2_name']}")
    elif static_te_2_to_1 > static_te_1_to_2:
        print(f"\nStronger direction: {config['data']['series2_name']} → {config['data']['series1_name']}")
    else:
        print("\nBidirectional causality (similar transfer entropy)")
    
    if config['model']['compute_rolling']:
        te_df = compute_rolling_te(
            series1, series2,
            config['model']['rolling_window'],
            config['model']['k'],
            config['model']['bins'],
            config
        )
        
        print(f"\nRolling Transfer Entropy Statistics:")
        print(f"Mean {te_df.columns[0]}: {te_df.iloc[:, 0].mean():.4f}")
        print(f"Mean {te_df.columns[1]}: {te_df.iloc[:, 1].mean():.4f}")
        
        create_visualizations(
            series1, series2, te_df,
            {'te_1_to_2': static_te_1_to_2, 'te_2_to_1': static_te_2_to_1},
            config
        )
    else:
        create_visualizations(
            series1, series2, pd.DataFrame(),
            {'te_1_to_2': static_te_1_to_2, 'te_2_to_1': static_te_2_to_1},
            config
        )
    
    print("✓ Transfer entropy analysis complete")


if __name__ == "__main__":
    main()

