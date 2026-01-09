#!/usr/bin/env python3
"""
Bollinger Bands for Time Series Analysis
Technical indicator using moving averages and standard deviations.
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
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)


def calculate_bollinger_bands(data: pd.Series, window: int, num_std: float):
    """Calculate Bollinger Bands."""
    df = pd.DataFrame({"value": data})
    df["MA"] = df["value"].rolling(window=window).mean()
    df["std"] = df["value"].rolling(window=window).std()
    df["upper"] = df["MA"] + (df["std"] * num_std)
    df["lower"] = df["MA"] - (df["std"] * num_std)
    return df


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data using consolidated loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value")
    )
    
    print(f"Loaded {len(series)} data points")
    
    # Calculate Bollinger Bands
    print("\nCalculating Bollinger Bands...")
    window = config["model"]["window"]
    num_std = config["model"]["num_std"]
    
    df_bb = calculate_bollinger_bands(series, window, num_std)
    
    # Create visualization
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=tuple(config.get("plotting", {}).get("figure_size", [12, 6])))
    
    ax.plot(
        series.index,
        series.values,
        "k-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label="Price",
    )
    ax.plot(
        df_bb.index,
        df_bb["MA"].values,
        "b-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        label="Moving Average",
    )
    ax.fill_between(
        df_bb.index,
        df_bb["lower"].values,
        df_bb["upper"].values,
        color="b",
        alpha=0.2,
        label=f"Bollinger Bands ({num_std}σ)",
    )
    
    ax.set_title(config.get("plot_titles", {}).get("bollinger_bands", "Bollinger Bands Analysis"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if config.get("output", {}).get("save_plots", True):
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "bollinger_bands.png", dpi=300)
        print(f"Plot saved to: {output_dir / 'bollinger_bands.png'}")
    
    print("\n Bollinger Bands analysis complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
