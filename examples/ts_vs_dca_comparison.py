#!/usr/bin/env python3
"""
Time Series vs Decline Curve Analysis Comparison

Complete example comparing time series forecasting methods against
traditional decline curve analysis (DCA) models.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import signalplot
from pipelines import ForecastingPipeline, register_model
from models.dca import ArpsExponential, ArpsHyperbolic, ArpsHarmonic

# Apply SignalPlot's clean defaults
signalplot.apply()


def main():
    """Run comparison example."""

    # Register DCA models
    register_model("Arps Exponential", lambda: ArpsExponential())
    register_model("Arps Hyperbolic", lambda: ArpsHyperbolic())
    register_model("Arps Harmonic", lambda: ArpsHarmonic())

    # Initialize pipeline
    data_path = Path(__file__).parent.parent / "data" / "production" / "well_production.csv"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure production data exists. See data/production/README.md")
        return

    pipeline = ForecastingPipeline(
        data_path=data_path,
        target_column="oil_rate",
        forecast_horizon=12,  # 12 months
        train_size=0.8,  # 80% for training
    )

    # Load data
    pipeline.load_data()
    print(f"Loaded {len(pipeline.train_data)} training points")

    # Add DCA models
    pipeline.add_model_from_registry("Arps Exponential")
    pipeline.add_model_from_registry("Arps Hyperbolic")
    pipeline.add_model_from_registry("Arps Harmonic")

    # Run all models
    print("\nRunning models...")
    results = pipeline.run_all()

    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    comparison = pipeline.compare_models(results)
    print(comparison.to_string())

    # Find best model
    from evaluation import ModelComparison

    comp = ModelComparison()
    for result in results.values():
        comp.add_result(result)

    best = comp.get_best_model("RMSE")
    if best:
        print(f"\nBest model (RMSE): {best.name}")
        if best.metrics:
            print(f"  RMSE: {best.metrics.rmse:.4f}")
            print(f"  MAE:  {best.metrics.mae:.4f}")
            print(f"  MAPE: {best.metrics.mape:.2f}%")
            print(f"  R²:   {best.metrics.r2:.4f}")

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical data
    ax.plot(
        pipeline.train_data.index,
        pipeline.train_data.values,
        "k-",
        linewidth=2,
        label="Historical (Train)",
        alpha=0.7,
    )

    # Plot test data if available
    if pipeline.test_data is not None:
        ax.plot(
            pipeline.test_data.index,
            pipeline.test_data.values,
            "g-",
            linewidth=2,
            label="Actual (Test)",
            alpha=0.7,
        )

    # Plot forecasts
    colors = ["r", "b", "orange", "purple", "brown"]
    for i, (name, result) in enumerate(results.items()):
        ax.plot(
            result.forecast.index,
            result.forecast.values,
            "--",
            linewidth=1.5,
            label=f"{name} Forecast",
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Production Rate (bbl/day)")
    ax.set_title("Time Series vs DCA Forecast Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Save plot
    output_dir = Path(__file__).parent.parent / "outputs" / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ts_vs_dca_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nPlot saved to: {output_path}")

    # Save results
    pipeline.save_results(results, output_dir, prefix="well_production")

    plt.show()


if __name__ == "__main__":
    main()

