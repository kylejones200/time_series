#!/usr/bin/env python3
"""
Kalman Filters: State Space Models
Kalman filtering and smoothing for time series analysis.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def load_data(config):
    """Load time series data."""
    # Use consolidated loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value")
    )
    return series.values


def create_kalman_filter(config):
    """Create Kalman filter based on config."""
    dim_x = config["model"]["state_dimension"]
    dim_z = config["model"]["measurement_dimension"]
    
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    
    kf.x = np.array(config["model"].get("initial_state", [0.0, 0.0]))
    kf.F = np.array(config["model"].get("state_transition", [[1.0, 1.0], [0.0, 1.0]]))
    kf.H = np.array(config["model"].get("measurement_matrix", [[1.0, 0.0]]))
    kf.P = np.eye(dim_x) * config["model"].get("initial_covariance", 1000.0)
    kf.R = config["model"].get("measurement_noise", 5.0)
    
    q = Q_discrete_white_noise(
        dim=dim_x,
        dt=config["model"].get("dt", 1.0),
        var=config["model"].get("process_noise", 0.1),
    )
    kf.Q = q
    
    return kf


def apply_kalman_filter(kf, measurements):
    """Apply Kalman filter to measurements."""
    estimates = []
    for measurement in measurements:
        kf.predict()
        kf.update(measurement)
        estimates.append(kf.x.copy())
    return np.array(estimates)


def create_visualizations(measurements, estimates, config, script_dir):
    """Generate clean visualizations."""
    fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))
    
    ax.plot(
        measurements,
        "k-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label="Measurements",
    )
    
    ax.plot(
        estimates[:, 0],
        "r--",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        label="Kalman Filter Estimate",
    )
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Kalman Filter State Estimation")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / "kalman_filter.png", dpi=300)
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data
    measurements = load_data(config)
    print(f"Loaded {len(measurements)} measurements")
    
    # Create and apply Kalman filter
    print("\nApplying Kalman filter...")
    kf = create_kalman_filter(config)
    estimates = apply_kalman_filter(kf, measurements)
    
    # Create visualizations
    print("\nCreating visualization...")
    create_visualizations(measurements, estimates, config, script_dir)
    
    print("\n Kalman filter analysis complete")


if __name__ == "__main__":
    main()
