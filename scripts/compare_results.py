#!/usr/bin/env python3
"""
Compare forecasting results across multiple templates.

Generates comparison plots and summary tables.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import save_plot


def load_forecast_results(output_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load forecast results from multiple template outputs."""
    results = {}
    
    for template_dir in output_dir.parent.glob("*_Python"):
        forecast_file = template_dir / "outputs" / "forecast.csv"
        if forecast_file.exists():
            try:
                df = pd.read_csv(forecast_file, encoding="utf-8")
                if "date" in df.columns and "forecast" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    results[template_dir.name] = df.set_index("date")
            except Exception as e:
                print(f"Warning: Could not load {forecast_file}: {e}")
    
    return results


def compare_forecasts(results: Dict[str, pd.DataFrame], actual: pd.Series, output_path: Path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Forecasts overlay
    axes[0].plot(actual.index, actual.values, "k-", lw=2, label="Actual", alpha=0.8)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, forecast_df), color in zip(results.items(), colors):
        if "forecast" in forecast_df.columns:
            axes[0].plot(
                forecast_df.index,
                forecast_df["forecast"],
                "--",
                lw=1.5,
                label=name.replace("_Python", ""),
                color=color,
                alpha=0.7
            )
    
    axes[0].set_title("Forecast Comparison")
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="best", ncol=2)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Errors
    for (name, forecast_df), color in zip(results.items(), colors):
        if "forecast" in forecast_df.columns:
            aligned = forecast_df["forecast"].reindex(actual.index)
            errors = (actual - aligned).dropna()
            if len(errors) > 0:
                axes[1].plot(errors.index, errors.values, "-", lw=1, label=name.replace("_Python", ""), color=color, alpha=0.7)
    
    axes[1].axhline(0, color="k", linestyle="--", alpha=0.3)
    axes[1].set_title("Forecast Errors")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Error")
    axes[1].legend(loc="best", ncol=2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, output_path / "forecast_comparison.png", dpi=300)
    plt.close(fig)


def generate_summary_table(results: Dict[str, pd.DataFrame], actual: pd.Series, output_path: Path):
    """Generate summary comparison table."""
    summary_data = []
    
    for name, forecast_df in results.items():
        if "forecast" in forecast_df.columns:
            aligned = forecast_df["forecast"].reindex(actual.index).dropna()
            actual_aligned = actual.reindex(aligned.index).dropna()
            
            if len(aligned) > 0 and len(actual_aligned) > 0:
                common_idx = aligned.index.intersection(actual_aligned.index)
                if len(common_idx) > 0:
                    errors = actual_aligned.loc[common_idx] - aligned.loc[common_idx]
                    
                    mae = errors.abs().mean()
                    rmse = np.sqrt((errors ** 2).mean())
                    mape = (errors.abs() / actual_aligned.loc[common_idx].abs()).mean() * 100
                    
                    summary_data.append({
                        "Model": name.replace("_Python", ""),
                        "MAE": mae,
                        "RMSE": rmse,
                        "MAPE": f"{mape:.2f}%",
                        "Points": len(common_idx)
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("RMSE")
        summary_df.to_csv(output_path / "comparison_summary.csv", index=False)
        
        print("\n Model Comparison Summary:")
        print("=" * 70)
        print(summary_df.to_string(index=False))
        print(f"\n Summary saved to: {output_path / 'comparison_summary.csv'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare forecasting results")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--actual", help="Path to actual values CSV (optional)")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    
    # Load results
    results = load_forecast_results(output_path)
    
    if not results:
        print("No forecast results found. Run some templates first.")
        sys.exit(1)
    
    print(f"Found {len(results)} forecast results")
    
    # Load actual if provided
    if args.actual:
        actual = pd.read_csv(args.actual, encoding="utf-8")
        actual["date"] = pd.to_datetime(actual["date"])
        actual = actual.set_index("date")["value"]
    else:
        print("️  No actual values provided. Skipping comparison plots.")
        actual = None
    
    # Generate comparison
    if actual is not None:
        compare_forecasts(results, actual, output_path)
        generate_summary_table(results, actual, output_path)
    else:
        print(" To generate comparison plots, provide --actual path to your test data")

