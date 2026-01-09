#!/usr/bin/env python3
"""
Orbit: Bayesian Time Series Forecasting
Bayesian structural time series models for forecasting.
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

from orbit.models import DLT, KTR, LGT


def create_model(config: dict):
    """Create Orbit model based on config."""
    model_map = {
        "DLT": DLT,
        "KTR": KTR,
        "LGT": LGT,
    }
    
    model_class = model_map[config["model"]["type"]]
    model_params = {
        "response_col": "value",
        "date_col": "date",
        **config["model"].get("params", {}),
    }
    
    return model_class(**model_params)


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data using consolidated loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"]["date_col"],
        value_column=config["data"]["value_col"]
    )
    
    # Convert to DataFrame format expected by Orbit
    df = pd.DataFrame({
        "date": series.index,
        "value": series.values
    })
    
    print(f"Loaded {len(df)} data points")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Create and fit model
    print(f"\nCreating {config['model']['type']} model...")
    model = create_model(config)
    
    print("Fitting model...")
    model.fit(df=df)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(df=df)
    
    # Create visualization
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))
    
    ax.plot(
        df["date"],
        df["value"],
        c=config.get("plotting", {}).get("style", {}).get("colors", {}).get("primary", "k"),
        linewidth=config.get("plotting", {}).get("style", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("style", {}).get("alpha", 0.8),
        label="Actual",
    )
    
    ax.plot(
        predictions["date"],
        predictions["prediction"],
        c=config.get("plotting", {}).get("style", {}).get("colors", {}).get("secondary", "r"),
        linewidth=config.get("plotting", {}).get("style", {}).get("linewidth", 1.5),
        label="Forecast",
    )
    
    # Add confidence intervals if available
    pred_lower = predictions.get("prediction_5") or predictions.get("prediction_lower")
    pred_upper = predictions.get("prediction_95") or predictions.get("prediction_upper")
    
    if pred_lower is not None and pred_upper is not None:
        ax.fill_between(
            predictions["date"],
            pred_lower,
            pred_upper,
            alpha=0.2,
            color=config.get("plotting", {}).get("style", {}).get("colors", {}).get("secondary", "r"),
            label="95% CI",
        )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"Orbit {config['model']['type']} Forecast")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot using consolidated utility
    if config.get("output", {}).get("save_plots", True):
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "orbit_forecast.png", dpi=300)
        print(f"Plot saved to: {output_dir / 'orbit_forecast.png'}")
    
    print("\n Orbit forecasting complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
