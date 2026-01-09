#!/usr/bin/env python3
"""
Transfer Entropy for Causal Inference
Information-theoretic causal inference using transfer entropy.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

import pyinform.transferentropy as te
from tqdm import tqdm


def discretize(series: pd.Series, bins: int):
    """Discretize continuous series into bins."""
    return pd.qcut(series.rank(method="first"), bins, labels=False).astype(int)


def compute_transfer_entropy(source: pd.Series, target: pd.Series, k: int, bins: int):
    """Compute transfer entropy between two series."""
    source_disc = discretize(source, bins).values.reshape(1, -1)
    target_disc = discretize(target, bins).values.reshape(1, -1)
    
    try:
        te_value = te.transfer_entropy(source_disc, target_disc, k=k)
        return te_value if not np.isnan(te_value) else 0.0
    except Exception:
        return 0.0


def compute_rolling_te(series1: pd.Series, series2: pd.Series, window: int, k: int, bins: int, config: dict):
    """Compute rolling transfer entropy."""
    te_1_to_2 = []
    te_2_to_1 = []
    dates = []
    
    for i in tqdm(range(window, len(series1)), desc="Computing rolling TE"):
        sub1 = series1.iloc[i - window : i]
        sub2 = series2.iloc[i - window : i]
        
        te_12 = compute_transfer_entropy(sub1, sub2, k, bins)
        te_21 = compute_transfer_entropy(sub2, sub1, k, bins)
        
        te_1_to_2.append(te_12)
        te_2_to_1.append(te_21)
        dates.append(series1.index[i])
    
    return pd.DataFrame(
        {
            f'{config["data"]["series1_name"]} → {config["data"]["series2_name"]}': te_1_to_2,
            f'{config["data"]["series2_name"]} → {config["data"]["series1_name"]}': te_2_to_1,
        },
        index=pd.to_datetime(dates),
    )


def create_visualizations(series1: pd.Series, series2: pd.Series, te_df: pd.DataFrame, static_te: dict, config: dict, script_dir: Path):
    """Generate visualizations for transfer entropy."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(
        series1.index,
        series1.values,
        "k-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label=config["data"]["series1_name"],
    )
    axes[0, 0].set_title(f"Time Series: {config['data']['series1_name']}")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(
        series2.index,
        series2.values,
        "b-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label=config["data"]["series2_name"],
    )
    axes[0, 1].set_title(f"Time Series: {config['data']['series2_name']}")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(
        te_df.index,
        te_df.iloc[:, 0],
        "r-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label=te_df.columns[0],
    )
    axes[1, 0].set_title("Rolling Transfer Entropy")
    axes[1, 0].set_ylabel("Transfer Entropy")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].legend(loc="best")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(
        te_df.index,
        te_df.iloc[:, 1],
        "g-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label=te_df.columns[1],
    )
    axes[1, 1].set_title("Rolling Transfer Entropy (Reverse)")
    axes[1, 1].set_ylabel("Transfer Entropy")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].legend(loc="best")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / "transfer_entropy.png", dpi=300)
    print(f"Plot saved to: {output_dir / 'transfer_entropy.png'}")


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data - Transfer entropy requires two series
    data_path = script_dir.parent / "data" / config["data"]["input_file"]
    df = pd.read_csv(data_path)
    
    series1_col = config["data"]["series1_col"]
    series2_col = config["data"]["series2_col"]
    date_col = config["data"].get("date_col", "date")
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    series1 = df[series1_col].dropna()
    series2 = df[series2_col].dropna()
    
    # Align series
    common_index = series1.index.intersection(series2.index)
    series1 = series1.loc[common_index]
    series2 = series2.loc[common_index]
    
    print(f"Loaded {len(series1)} data points for both series")
    
    # Compute static transfer entropy
    k = config["model"]["k"]
    bins = config["model"]["bins"]
    
    print("\nComputing static transfer entropy...")
    te_1_to_2 = compute_transfer_entropy(series1, series2, k, bins)
    te_2_to_1 = compute_transfer_entropy(series2, series1, k, bins)
    
    static_te = {
        f'{config["data"]["series1_name"]} → {config["data"]["series2_name"]}': te_1_to_2,
        f'{config["data"]["series2_name"]} → {config["data"]["series1_name"]}': te_2_to_1,
    }
    
    print(f"\nStatic Transfer Entropy:")
    for key, value in static_te.items():
        print(f"  {key}: {value:.4f}")
    
    # Compute rolling transfer entropy if configured
    if config["model"].get("rolling_te", True):
        window = config["model"].get("rolling_window", 100)
        print(f"\nComputing rolling transfer entropy (window={window})...")
        te_df = compute_rolling_te(series1, series2, window, k, bins, config)
    else:
        te_df = pd.DataFrame(static_te, index=[series1.index[0]])
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(series1, series2, te_df, static_te, config, script_dir)
    
    print("\n Transfer entropy analysis complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
