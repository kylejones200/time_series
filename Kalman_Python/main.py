#!/usr/bin/env python3
"""
Kalman Filters: State Space Models
Kalman filtering and smoothing for time series analysis.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src import BaseTemplate
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def create_kalman_filter(config: dict) -> KalmanFilter:
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


def apply_kalman_filter(kf: KalmanFilter, measurements: np.ndarray) -> np.ndarray:
    """Apply Kalman filter to measurements."""
    estimates = []
    for measurement in measurements:
        kf.predict()
        kf.update(measurement)
        estimates.append(kf.x.copy())
    return np.array(estimates)


def main():
    """Main execution function."""
    template = BaseTemplate(config_path="config.yaml", script_dir=Path(__file__).parent)
    config = template.config
    series = template.load_data()
    measurements = series.values

    print(f"Loaded {len(measurements)} measurements")

    print("\nApplying Kalman filter...")
    kf = create_kalman_filter(config)
    estimates = apply_kalman_filter(kf, measurements)

    print("\nCreating visualization...")
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
    plt.tight_layout()

    plot_path = template.save_plot(fig, "kalman_filter.png")
    print(f"Plot saved to: {plot_path}")

    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)

    print("\n Kalman filter analysis complete")


if __name__ == "__main__":
    main()
