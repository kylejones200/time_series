#!/usr/bin/env python3
"""
Reference forecasting script.

This is the reference implementation for all forecasting examples.
It demonstrates the standard workflow:
1. Load time series data
2. Split into train/test
3. Fit model
4. Generate forecast
5. Evaluate
6. Save results (plot + CSV)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import from consolidated src module (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ARIMAModel,
    Evaluator,
    create_forecast_plot,
    save_plot,
)


def main():
    """Run reference forecasting workflow using consolidated utilities."""
    
    # Load configuration
    config = load_config()
    
    # 1. Load data
    print(f"Loading data from {config['data']['input_file']}")
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"]["date_column"],
        value_column=config["data"]["value_column"]
    )
    print(f"Loaded {len(series)} data points")
    print(f"Date range: {series.index.min()} to {series.index.max()}")
    
    # 2. Split into train/test
    test_size = config["evaluation"]["test_size"]
    evaluator = Evaluator(test_size=test_size)
    train, test = evaluator.split(series)
    print(f"\nTrain: {len(train)} points")
    print(f"Test: {len(test)} points")
    
    # 3. Fit model
    print(f"\nFitting {config['model']['name']} model...")
    model = ARIMAModel(**{k: v for k, v in config["model"].items() if k != "name"})
    model.fit(train)
    order = model.get_order()
    print(f"Best ARIMA order: {order}")
    
    # 4. Generate forecast
    print(f"\nGenerating forecast for {len(test)} periods...")
    forecast, conf_int = model.forecast(n_periods=len(test), return_conf_int=True)
    print(f"Forecast generated: {len(forecast)} points")
    
    # 5. Evaluate
    print("\nEvaluating forecast...")
    metrics = evaluator.evaluate(forecast, test)
    metric_name = config["evaluation"]["metric"]
    metric_value = metrics[metric_name]
    print(f"{metric_name}: {metric_value:.4f}")
    print(f"Evaluation points: {metrics['n_points']}")
    
    # 6. Create plot using consolidated plotting utility
    print("\nCreating plot...")
    fig, ax = create_forecast_plot(
        train=train,
        test=test,
        forecast=forecast,
        conf_int=conf_int,
        figsize=tuple(config["plotting"]["figure_size"]),
        title=f"ARIMA{order} Forecast ({metric_name}: {metric_value:.4f})",
    )
    
    # 7. Save results using consolidated utilities
    script_dir = Path(__file__).parent
    from src import get_output_dir, ensure_output_dir
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    
    # Save plot
    plot_path = save_plot(
        fig,
        output_dir / config["output"]["plot_name"],
        dpi=config["output"]["dpi"]
    )
    print(f"Plot saved to: {plot_path}")
    
    # Save forecast CSV
    forecast_df = pd.DataFrame({
        "date": forecast.index,
        "forecast": forecast.values,
        "lower": conf_int["lower"].values,
        "upper": conf_int["upper"].values,
    })
    csv_path = output_dir / config["output"]["forecast_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(csv_path, index=False)
    print(f"Forecast CSV saved to: {csv_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / config["output"]["metrics_csv"]
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics CSV saved to: {metrics_path}")
    
    print("\n Reference forecast complete!")
    
    if config["plotting"]["show_plot"]:
        plt.show()


if __name__ == "__main__":
    main()

