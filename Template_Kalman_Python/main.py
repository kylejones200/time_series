#!/usr/bin/env python3
"""
Kalman Filters: State Space Models
Kalman filtering and smoothing for time series analysis.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import util
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


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


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(config):
    """Load time series data."""
    data_path = Path(__file__).parent.parent / 'data' / Path(config['data']['input_file']).name
    df = pd.read_csv(data_path)
    
    date_col = config['data'].get('date_col')
    value_col = config['data']['value_col']
    
    [df.__setitem__(date_col, pd.to_datetime(df[date_col]))
     for _ in [None] if date_col]
    
    [df.set_index(date_col, inplace=True) for _ in [None] if date_col]
    
    return df[value_col].values


def create_kalman_filter(config):
    """Create Kalman filter based on config."""
    dim_x = config['model']['state_dimension']
    dim_z = config['model']['measurement_dimension']
    
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    
    kf.x = np.array(config['model'].get('initial_state', [0.0, 0.0]))
    kf.F = np.array(config['model'].get('state_transition', [[1., 1.], [0., 1.]]))
    kf.H = np.array(config['model'].get('measurement_matrix', [[1., 0.]]))
    kf.P = np.eye(dim_x) * config['model'].get('initial_covariance', 1000.0)
    kf.R = config['model'].get('measurement_noise', 5.0)
    
    q = Q_discrete_white_noise(
        dim=dim_x,
        dt=config['model'].get('dt', 1.0),
        var=config['model'].get('process_noise', 0.1)
    )
    kf.Q = q
    
    return kf


def apply_kalman_filter(kf, measurements):
    """Apply Kalman filter to measurements."""
    estimates = []
    
    [kf.predict() or kf.update(measurement) or estimates.append(kf.x.copy())
     for measurement in measurements]
    
    return np.array(estimates)


def create_visualizations(measurements, estimates, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = setup_figure(config)
    
    ax.plot(measurements,
            c=config['plotting']['style']['colors']['primary'],
            linewidth=config['plotting']['style']['linewidth'],
            alpha=config['plotting']['style']['alpha'],
            label='Measurements')
    
    ax.plot(estimates[:, 0],
            c=config['plotting']['style']['colors']['secondary'],
            linewidth=config['plotting']['style']['linewidth'],
            label='Kalman Filter Estimate')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'kalman_filter.png', config)
     for _ in [None] if config['output']['save_plots']]
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    measurements = load_data(config)
    kf = create_kalman_filter(config)
    estimates = apply_kalman_filter(kf, measurements)
    create_visualizations(measurements, estimates, config)
    
    print("✓ Kalman filtering complete")


if __name__ == "__main__":
    main()

