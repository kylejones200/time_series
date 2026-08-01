#!/usr/bin/env python3
"""
Nixtla: StatsForecast
Fast statistical forecasting with Nixtla's StatsForecast library.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    AutoCES,
    DynamicOptimizedTheta,
    SeasonalNaive,
)

# Try to import HierarchicalForecast (optional dependency)
try:
    from hierarchicalforecast.core import HierarchicalForecast
    from hierarchicalforecast.methods import BottomUp
    try:
        from hierarchicalforecast.utils import aggregate
    except ImportError:
        # Try alternative import path
        try:
            from hierarchicalforecast.utils import agg_series as aggregate
        except ImportError:
            aggregate = None
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    aggregate = None


def prepare_data(data: pd.Series, config: dict) -> pd.DataFrame:
    """Prepare data for StatsForecast (requires 'ds' and 'y' columns, 'unique_id' for multiple series)."""
    df = pd.DataFrame(
        {
            "unique_id": config["data"].get("unique_id", "series1"),
            "ds": data.index,
            "y": data.values,
        }
    )
    return df


def create_models(config: dict) -> list:
    """Create list of models for StatsForecast."""
    model_map = {
        "AutoARIMA": AutoARIMA,
        "AutoETS": AutoETS,
        "AutoTheta": AutoTheta,
        "AutoCES": AutoCES,
        "DynamicOptimizedTheta": DynamicOptimizedTheta,
        "SeasonalNaive": SeasonalNaive,
    }
    
    model_names = config["model"].get("models", ["AutoARIMA"])
    return [model_map[name]() for name in model_names]


def fit_and_predict(models: list, df: pd.DataFrame, config: dict) -> tuple:
    """Fit models and generate predictions."""
    sf = StatsForecast(
        models=models,
        freq=config["model"].get("freq", "D"),
        n_jobs=config["model"].get("n_jobs", 1),
    )
    
    sf.fit(df)
    
    forecast_horizon = config["model"]["forecast_horizon"]
    forecasts = sf.predict(h=forecast_horizon)
    
    return sf, forecasts


def create_visualizations(df: pd.DataFrame, forecasts: pd.DataFrame, config: dict, script_dir: Path) -> None:
    """Generate clean visualizations."""
    fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))
    
    # Plot historical data
    ax.plot(
        df["ds"],
        df["y"],
        c=config.get("plotting", {}).get("style", {}).get("colors", {}).get("primary", "k"),
        linewidth=config.get("plotting", {}).get("style", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("style", {}).get("alpha", 0.8),
        label="Historical",
    )
    
    # Plot forecasts from different models
    model_cols = [col for col in forecasts.columns if col not in ["ds", "unique_id"]]
    
    for col in model_cols[:3]:  # Limit to 3 models for clarity
        ax.plot(
            forecasts["ds"],
            forecasts[col],
            linewidth=config.get("plotting", {}).get("style", {}).get("linewidth", 1.5),
            label=col,
            alpha=0.8,
        )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("StatsForecast - Multiple Models Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if config.get("output", {}).get("save_plots", True):
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "nixtla_forecast.png", dpi=300)
        print(f"Plot saved to: {output_dir / 'nixtla_forecast.png'}")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


def run_hierarchical_forecast(df: pd.DataFrame, config: dict, script_dir: Path):
    """Run hierarchical forecasting if hierarchy is defined."""
    if not HIERARCHICAL_AVAILABLE:
        print("HierarchicalForecast not available. Install with: pip install hierarchicalforecast")
        return
    
    hierarchy = config.get("hierarchical", {}).get("structure")
    if not hierarchy:
        return None
    
    print("\nRunning HierarchicalForecast...")
    
    # Aggregate series according to hierarchy
    if aggregate is None:
        print("Warning: aggregate function not available. Using data as-is.")
        Y_df = df.copy()
        S_df = None
    else:
        Y_df, S_df = aggregate(df, hierarchy)
    
    # Create base model
    base_model = AutoARIMA()
    
    # Create hierarchical forecast model
    hf_model = HierarchicalForecast(
        models=[base_model],
        reconciliation=[BottomUp()],
    )
    
    # Fit
    freq = config["model"].get("freq", "D")
    hf_model.fit(Y_df=Y_df, S_df=S_df, freq=freq)
    
    # Forecast
    forecast_horizon = config["model"]["forecast_horizon"]
    forecasts = hf_model.predict(h=forecast_horizon)
    
    print(f"Generated hierarchical forecasts for {len(forecasts)} series")
    
    # Visualize hierarchy levels
    if config.get("hierarchical", {}).get("visualize", True):
        levels = config.get("hierarchical", {}).get("visualize_levels", list(hierarchy.keys())[:3])
        
        fig, axes = plt.subplots(len(levels), 1, figsize=(12, 4 * len(levels)))
        if len(levels) == 1:
            axes = [axes]
        
        for i, level in enumerate(levels):
            if level in Y_df["unique_id"].values:
                historical = Y_df[Y_df["unique_id"] == level]
                forecast = forecasts[forecasts["unique_id"] == level]
                
                axes[i].plot(historical["ds"], historical["y"], label="Historical", linewidth=1.5)
                axes[i].plot(forecast["ds"], forecast["BottomUp"], label="Forecast", color="red", linewidth=1.5)
                axes[i].set_title(f"Hierarchical Forecast: {level}")
                axes[i].set_xlabel("Date")
                axes[i].set_ylabel("Value")
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "hierarchical_forecast.png", dpi=300)
        print(f"Hierarchical forecast plot saved to: {output_dir / 'hierarchical_forecast.png'}")
    
    return forecasts


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Check if hierarchical forecasting is requested
    use_hierarchical = config.get("hierarchical", {}).get("enabled", False)
    
    if use_hierarchical and HIERARCHICAL_AVAILABLE:
        # Load data for hierarchical forecasting
        data = load_time_series(
            config["data"]["input_file"],
            date_column=config["data"].get("date_col", "date"),
            value_column=config["data"].get("value_col", "value")
        )
        
        # Prepare data
        df = prepare_data(data, config)
        
        # Add hierarchy information if not present
        if "unique_id" not in df.columns or df["unique_id"].nunique() == 1:
            # Create simple hierarchy for demonstration
            hierarchy = config.get("hierarchical", {}).get("structure")
            if hierarchy:
                # This is a simplified example - in practice, you'd have proper hierarchy data
                print("Note: HierarchicalForecast requires proper hierarchy structure in data")
        
        # Run hierarchical forecast
        forecasts = run_hierarchical_forecast(df, config, script_dir)
        
        if forecasts is not None:
            print("\n HierarchicalForecast complete")
            return
    
    # Standard StatsForecast
    # Load data using consolidated loader
    data = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value")
    )
    
    print(f"Loaded {len(data)} data points")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Prepare data for StatsForecast
    df = prepare_data(data, config)
    
    # Create and fit models
    print("\nFitting StatsForecast models...")
    models = create_models(config)
    sf, forecasts = fit_and_predict(models, df, config)
    
    print(f"Generated forecasts for {len(forecasts)} periods")
    
    # Create visualizations
    print("\nCreating visualization...")
    create_visualizations(df, forecasts, config, script_dir)
    
    print("\n Nixtla StatsForecast complete")


if __name__ == "__main__":
    main()
