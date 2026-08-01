#!/usr/bin/env python3
"""
Example Template Using Enhanced Plotting Configuration

This demonstrates how to use the enhanced config structure with
utils.plotting_utils for more granular control over plot styling.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import matplotlib.pyplot as plt

# Import enhanced plotting utilities
from utils.plotting_utils import (
    setup_figure,
    apply_plot_style,
    apply_legend,
    save_plot as save_plot_enhanced,
)

# Import standard utilities
from src.config import load_config
from src.loader import load_time_series
from src.utils import ensure_output_dir


def main():
    """
    Main execution function demonstrating enhanced config usage.
    
    This example shows:
    1. How to load config with enhanced plotting structure
    2. How to use setup_figure() for styled figures
    3. How to apply config colors and styling
    4. How to save plots with enhanced save_plot()
    """
    script_dir = Path(__file__).parent
    config = load_config("enhanced_config_example.yaml")
    
    print("Loading data...")
    # Load data using standard loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"]["date_col"],
        value_column=config["data"]["value_col"]
    )
    print(f"Loaded {len(series)} data points")
    
    # Create visualization with enhanced config
    print("\nCreating visualization with enhanced config...")
    fig, ax = setup_figure(config, nrows=1, ncols=1)
    
    # Get style configuration
    style = config["plotting"]["style"]
    
    # Plot data using config colors and styling
    ax.plot(
        series.index,
        series.values,
        color=style["colors"]["primary"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        label="Time Series Data"
    )
    
    # Add a trend line using secondary color
    # (This is just an example - you'd calculate actual trend)
    if len(series) > 0:
        ax.axhline(
            y=series.mean(),
            color=style["colors"]["secondary"],
            linewidth=style["linewidth"] * 0.8,
            alpha=style["alpha"] * 0.5,
            linestyle="--",
            label="Mean"
        )
    
    # Set labels and title
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Enhanced Config Example - Styled Plot", fontsize=14, fontweight="bold")
    
    # Apply legend with config settings
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    # Save plot using enhanced save_plot
    output_dir = ensure_output_dir(script_dir / "outputs")
    
    if config["output"].get("save_plots", True):
        output_path = output_dir / "enhanced_config_example.png"
        save_plot_enhanced(
            fig,
            output_path,
            config
        )
        print(f" Plot saved to: {output_path}")
    
    # Show or close plot based on config
    if config["plotting"].get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)
    
    print("\n Enhanced config example complete!")
    print("\nKey features demonstrated:")
    print("  • Config-driven spine control (top/right hidden)")
    print("  • Config-driven colors (primary/secondary)")
    print("  • Config-driven linewidth and alpha")
    print("  • Config-driven legend styling")
    print("  • Config-driven figure size and DPI")


if __name__ == "__main__":
    main()

