#!/usr/bin/env python3
"""
Copula Methods for Multivariate Time Series
Multivariate dependency modeling using copulas for time series forecasting.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
import scipy.stats as stats
from copulas.bivariate import Clayton, Gumbel, Frank
from copulas.multivariate import GaussianMultivariate

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def transform_to_uniform(data):
    """Transform data to uniform scale using rank transformation."""
    n = len(data)
    return stats.rankdata(data) / (n + 1)


def fit_copula(data, copula_type, config):
    """Fit copula to data."""
    u = transform_to_uniform(data.iloc[:, 0])
    v = transform_to_uniform(data.iloc[:, 1])
    
    copula_map = {
        'Clayton': lambda: Clayton(),
        'Gumbel': lambda: Gumbel(),
        'Frank': lambda: Frank(),
        'Gaussian': lambda: GaussianMultivariate(),
    }
    
    copula = copula_map.get(copula_type, copula_map['Clayton'])()
    
    if copula_type == 'Gaussian':
        copula.fit(pd.DataFrame({'u': u, 'v': v}))
    else:
        copula.fit(np.column_stack((u, v)))
    
    return copula, u, v


def simulate_from_copula(copula, n_samples, copula_type):
    """Simulate from fitted copula."""
    if copula_type == 'Gaussian':
        samples = copula.sample(n_samples).to_numpy()
    else:
        samples = copula.sample(n_samples)
    
    return samples


def transform_from_uniform(samples, data):
    """Transform uniform samples back to original scale."""
    forecast1 = np.quantile(data.iloc[:, 0], samples[:, 0])
    forecast2 = np.quantile(data.iloc[:, 1], samples[:, 1])
    return forecast1, forecast2


def create_visualizations(data, u, v, copula_samples, forecast1, forecast2, copula_type, config):
    """Generate visualizations for copula analysis."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for ax in axes.flatten():
        apply_plot_style(ax, {'plotting': config['plotting']})
    
    axes[0, 0].plot(data.index, data.iloc[:, 0].values,
                    'k-', linewidth=config['plotting']['linewidth'],
                    alpha=config['plotting']['alpha'], label=config['data']['series1_name'])
    axes[0, 0].plot(data.index, data.iloc[:, 1].values,
                    'r-', linewidth=config['plotting']['linewidth'],
                    alpha=config['plotting']['alpha'], label=config['data']['series2_name'])
    axes[0, 0].set_title('Original Time Series')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value')
    apply_legend(axes[0, 0], config['plotting']['legend'])
    
    axes[0, 1].scatter(u, v, alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
    axes[0, 1].set_title('Uniform Transformed Data')
    axes[0, 1].set_xlabel(f'{config["data"]["series1_name"]} (uniform)')
    axes[0, 1].set_ylabel(f'{config["data"]["series2_name"]} (uniform)')
    
    axes[1, 0].scatter(copula_samples[:, 0], copula_samples[:, 1],
                      alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
    axes[1, 0].set_title(f'{copula_type} Copula Samples')
    axes[1, 0].set_xlabel('u (uniform)')
    axes[1, 0].set_ylabel('v (uniform)')
    
    axes[1, 1].scatter(forecast1, forecast2,
                      alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
    axes[1, 1].set_title(f'{copula_type} Copula Forecast')
    axes[1, 1].set_xlabel(f'Forecasted {config["data"]["series1_name"]}')
    axes[1, 1].set_ylabel(f'Forecasted {config["data"]["series2_name"]}')
    
    plt.tight_layout()
    
    output_path = output_dir / "copula_analysis.png"
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
    
    common_index = df1.index.intersection(df2.index)
    data = pd.DataFrame({
        config['data']['series1_col']: df1.loc[common_index, config['data']['series1_col']],
        config['data']['series2_col']: df2.loc[common_index, config['data']['series2_col']]
    }, index=common_index)
    
    if config['data']['difference']:
        data = data.diff().dropna()
    
    copula, u, v = fit_copula(data, config['model']['copula_type'], config)
    
    print(f"\nCopula Analysis ({config['model']['copula_type']}):")
    print("=" * 70)
    print(f"Data points: {len(data)}")
    print(f"Copula type: {config['model']['copula_type']}")
    
    copula_samples = simulate_from_copula(
        copula,
        config['model']['n_samples'],
        config['model']['copula_type']
    )
    
    forecast1, forecast2 = transform_from_uniform(copula_samples, data)
    
    print(f"\nForecast Statistics:")
    print(f"{config['data']['series1_name']} forecast mean: {np.mean(forecast1):.4f}")
    print(f"{config['data']['series2_name']} forecast mean: {np.mean(forecast2):.4f}")
    print(f"Correlation in forecast: {np.corrcoef(forecast1, forecast2)[0, 1]:.4f}")
    
    create_visualizations(
        data, u, v, copula_samples, forecast1, forecast2,
        config['model']['copula_type'], config
    )
    
    print("✓ Copula analysis complete")


if __name__ == "__main__":
    main()

