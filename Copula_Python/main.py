#!/usr/bin/env python3
"""
Copula Methods for Multivariate Time Series
Multivariate dependency modeling using copulas for time series forecasting.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from copulas.bivariate import Clayton, Gumbel, Frank
from copulas.multivariate import GaussianMultivariate


def transform_to_uniform(data: np.ndarray):
    """Transform data to uniform scale using rank transformation."""
    n = len(data)
    return stats.rankdata(data) / (n + 1)


def fit_copula(data: pd.DataFrame, copula_type: str, config: dict):
    """Fit copula to data."""
    u = transform_to_uniform(data.iloc[:, 0])
    v = transform_to_uniform(data.iloc[:, 1])
    
    copula_map = {
        "Clayton": lambda: Clayton(),
        "Gumbel": lambda: Gumbel(),
        "Frank": lambda: Frank(),
        "Gaussian": lambda: GaussianMultivariate(),
    }
    
    copula = copula_map.get(copula_type, copula_map["Clayton"])()
    
    if copula_type == "Gaussian":
        copula.fit(pd.DataFrame({"u": u, "v": v}))
    else:
        copula.fit(np.column_stack((u, v)))
    
    return copula, u, v


def simulate_from_copula(copula, n_samples: int, copula_type: str):
    """Simulate from fitted copula."""
    if copula_type == "Gaussian":
        samples = copula.sample(n_samples).to_numpy()
    else:
        samples = copula.sample(n_samples)
    return samples


def transform_from_uniform(samples: np.ndarray, data: pd.DataFrame):
    """Transform uniform samples back to original scale."""
    forecast1 = np.quantile(data.iloc[:, 0], samples[:, 0])
    forecast2 = np.quantile(data.iloc[:, 1], samples[:, 1])
    return forecast1, forecast2


def create_visualizations(
    data: pd.DataFrame, u: np.ndarray, v: np.ndarray, copula_samples: np.ndarray,
    forecast1: np.ndarray, forecast2: np.ndarray, copula_type: str, config: dict, script_dir: Path
):
    """Generate visualizations for copula analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original data scatter
    axes[0, 0].scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.6, s=20)
    axes[0, 0].set_xlabel("Series 1")
    axes[0, 0].set_ylabel("Series 2")
    axes[0, 0].set_title("Original Data")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Uniform marginals
    axes[0, 1].scatter(u, v, alpha=0.6, s=20, c="red")
    axes[0, 1].set_xlabel("u (Uniform Series 1)")
    axes[0, 1].set_ylabel("v (Uniform Series 2)")
    axes[0, 1].set_title("Uniform Marginals")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Copula samples
    axes[1, 0].scatter(copula_samples[:, 0], copula_samples[:, 1], alpha=0.6, s=20, c="green")
    axes[1, 0].set_xlabel("u (Sample)")
    axes[1, 0].set_ylabel("v (Sample)")
    axes[1, 0].set_title(f"{copula_type} Copula Samples")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Forecasts
    axes[1, 1].scatter(forecast1, forecast2, alpha=0.6, s=20, c="blue")
    axes[1, 1].set_xlabel("Forecast Series 1")
    axes[1, 1].set_ylabel("Forecast Series 2")
    axes[1, 1].set_title(f"{copula_type} Forecast")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / f"copula_{copula_type.lower()}.png", dpi=300)
    print(f"Plot saved to: {output_dir / f'copula_{copula_type.lower()}.png'}")


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data - Copula requires multivariate data
    data_path = script_dir.parent / "data" / config["data"]["input_file"]
    df = pd.read_csv(data_path, encoding="utf-8")
    
    value_cols = config["data"]["value_cols"]
    data = df[value_cols].dropna()
    
    print(f"Loaded {len(data)} data points with {len(value_cols)} variables")
    
    copula_type = config["model"]["type"]
    
    # Fit copula
    print(f"\nFitting {copula_type} copula...")
    copula, u, v = fit_copula(data, copula_type, config)
    
    # Simulate from copula
    n_samples = config["model"].get("n_samples", 1000)
    print(f"Simulating {n_samples} samples from copula...")
    copula_samples = simulate_from_copula(copula, n_samples, copula_type)
    
    # Transform to original scale
    forecast1, forecast2 = transform_from_uniform(copula_samples, data)
    
    print(f"\nGenerated forecasts:")
    print(f"  Series 1: mean={forecast1.mean():.4f}, std={forecast1.std():.4f}")
    print(f"  Series 2: mean={forecast2.mean():.4f}, std={forecast2.std():.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(
        data, u, v, copula_samples, forecast1, forecast2, copula_type, config, script_dir
    )
    
    print("\n Copula analysis complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
