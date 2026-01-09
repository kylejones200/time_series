#!/usr/bin/env python3
"""
Unified CLI for time series forecasting.

Run any template from the root directory with a simple command.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import load_config, load_time_series


def list_templates():
    """List all available forecasting templates."""
    repo_root = Path(__file__).parent
    templates = sorted([d.name for d in repo_root.iterdir() if d.is_dir() and d.name.endswith("_Python")])
    
    print("Available Forecasting Templates:")
    print("=" * 70)
    
    categories = {
        "Classical": ["ARIMA", "ARAR", "BoxJenkins", "MovingAverage", "ExponentialSmoothing", "VAR", "Differencing", "Kalman"],
        "Bayesian": ["Bayesian", "BayesianChangePoint", "Orbit", "PyBSTS"],
        "Modern Libraries": ["Prophet", "Darts", "StatsForecast", "Greykite", "Merlion", "Autogluon", "PyCaret"],
        "Deep Learning": ["LSTM", "NBEATS", "TSAI", "BERT"],
        "Foundation Models": ["Chronos", "TimesFM", "LagLlama", "Sundial"],
        "Specialized": ["TSFresh", "tslearn", "Aeon", "STUMPY_PyOD", "CCM", "Copula", "RegimeSwitching", "TransferEntropy", "Volatility"],
        "Other": []
    }
    
    for category, keywords in categories.items():
        matching = [t for t in templates if any(kw in t for kw in keywords)]
        if matching:
            print(f"\n{category}:")
            for template in matching:
                print(f"  - {template}")
            templates = [t for t in templates if t not in matching]
    
    if templates:
        print(f"\nOther:")
        for template in templates:
            print(f"  - {template}")
    
    print(f"\nTotal: {len(repo_root.glob('*_Python'))} templates")


def run_template(template_name: str, data_path: Optional[str] = None, config_path: Optional[str] = None):
    """Run a specific forecasting template."""
    repo_root = Path(__file__).parent
    template_dir = repo_root / template_name
    
    if not template_dir.exists():
        print(f"Error: Template '{template_name}' not found.")
        print(f"\nAvailable templates:")
        list_templates()
        sys.exit(1)
    
    main_py = template_dir / "main.py"
    if not main_py.exists():
        print(f"Error: {main_py} not found.")
        sys.exit(1)
    
    # Change to template directory and run
    import subprocess
    import os
    
    original_dir = os.getcwd()
    try:
        os.chdir(template_dir)
        
        # Override config if provided
        env = os.environ.copy()
        if config_path:
            env["CONFIG_PATH"] = str(Path(config_path).resolve())
        if data_path:
            env["DATA_PATH"] = str(Path(data_path).resolve())
        
        result = subprocess.run([sys.executable, "main.py"], env=env)
        sys.exit(result.returncode)
    finally:
        os.chdir(original_dir)


def benchmark(data_path: str, templates: Optional[list] = None, output_dir: str = "outputs/benchmark"):
    """Run multiple templates and compare results."""
    from pathlib import Path
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import json
    
    repo_root = Path(__file__).parent
    
    if templates is None:
        # Use a curated set of fast, representative templates
        templates = [
            "ARIMA_Python",
            "Prophet_Python",
            "MovingAverage_Python",
            "ExponentialSmoothing_Python",
            "StatsForecast_Python",
        ]
    
    print(f"Benchmarking {len(templates)} templates on {data_path}")
    print("=" * 70)
    
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for template_name in templates:
        template_dir = repo_root / template_name
        if not template_dir.exists():
            print(f"️  Skipping {template_name} (not found)")
            continue
        
        print(f"\n Running {template_name}...")
        try:
            # This would need to be implemented to capture metrics
            # For now, just run the template
            run_template(template_name, data_path=data_path)
            print(f" {template_name} completed")
        except Exception as e:
            print(f" {template_name} failed: {e}")
            results.append({
                "template": template_name,
                "status": "failed",
                "error": str(e)
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / "benchmark_results.csv", index=False, encoding="utf-8")
    print(f"\n Results saved to {output_path / 'benchmark_results.csv'}")


def recommend(data_path: str):
    """Recommend templates based on data characteristics."""
    from src import load_time_series
    import pandas as pd
    import numpy as np
    
    print("Analyzing data characteristics...")
    
    try:
        series = load_time_series(data_path, date_column="date", value_column="value")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nExpected CSV format:")
        print("  date,value")
        print("  2020-01-01,100.5")
        sys.exit(1)
    
    n_points = len(series)
    has_trend = False
    has_seasonality = False
    
    # Simple heuristics
    if n_points > 50:
        # Check for trend
        first_half = series.iloc[:len(series)//2].mean()
        second_half = series.iloc[len(series)//2:].mean()
        has_trend = abs(second_half - first_half) / first_half > 0.1
        
        # Check for seasonality (simple autocorrelation check)
        if n_points > 24:
            autocorr = series.autocorr(lag=min(12, n_points//4))
            has_seasonality = abs(autocorr) > 0.3
    
    print(f"\nData Characteristics:")
    print(f"  Length: {n_points} points")
    print(f"  Trend: {'Yes' if has_trend else 'No'}")
    print(f"  Seasonality: {'Yes' if has_seasonality else 'No'}")
    
    print(f"\n Recommended Templates:")
    print("=" * 70)
    
    recommendations = []
    
    if n_points < 30:
        recommendations.extend([
            ("MovingAverage_Python", "Simple baseline for short series"),
            ("ExponentialSmoothing_Python", "Good for short series with trends"),
        ])
    elif n_points < 100:
        recommendations.extend([
            ("ARIMA_Python", "Classical method, good for medium series"),
            ("Prophet_Python", "Handles seasonality well"),
            ("ExponentialSmoothing_Python", "Good for trends and seasonality"),
        ])
    else:
        recommendations.extend([
            ("ARIMA_Python", "Classical method, interpretable"),
            ("Prophet_Python", "Excellent for seasonality"),
            ("StatsForecast_Python", "Fast and accurate"),
        ])
        
        if has_seasonality:
            recommendations.append(("Prophet_Python", "Best for seasonal data"))
        
        if n_points > 200:
            recommendations.extend([
                ("LSTM_Python", "Deep learning for long series"),
                ("Chronos_Python", "Foundation model, pre-trained"),
            ])
    
    for template, reason in recommendations:
        print(f"  • {template}")
        print(f"    {reason}")
    
    print(f"\n Tip: Run 'python forecast.py benchmark --data {data_path}' to compare multiple methods")


def validate_data(data_path: str):
    """Validate time series data format."""
    import pandas as pd
    
    print(f"Validating data: {data_path}")
    print("=" * 70)
    
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
        print(f" File loaded successfully")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check required columns
        required = ["date", "value"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"\n Missing required columns: {missing}")
            print(f"   Available columns: {list(df.columns)}")
            return False
        
        # Check date column
        try:
            dates = pd.to_datetime(df["date"])
            print(f" Date column valid")
            print(f"   Date range: {dates.min()} to {dates.max()}")
        except Exception as e:
            print(f" Date column invalid: {e}")
            return False
        
        # Check value column
        try:
            values = pd.to_numeric(df["value"], errors="coerce")
            n_missing = values.isna().sum()
            if n_missing > 0:
                print(f"️  Warning: {n_missing} missing values in 'value' column")
            else:
                print(f" Value column valid")
            print(f"   Value range: {values.min():.2f} to {values.max():.2f}")
        except Exception as e:
            print(f" Value column invalid: {e}")
            return False
        
        print(f"\n Data validation passed!")
        return True
        
    except FileNotFoundError:
        print(f" File not found: {data_path}")
        return False
    except Exception as e:
        print(f" Error: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified CLI for time series forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all templates
  python forecast.py list

  # Run a specific template
  python forecast.py run ARIMA_Python --data data/my_series.csv

  # Validate your data
  python forecast.py validate data/my_series.csv

  # Get template recommendations
  python forecast.py recommend data/my_series.csv

  # Benchmark multiple templates
  python forecast.py benchmark --data data/my_series.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List templates
    list_parser = subparsers.add_parser("list", help="List all available templates")
    
    # Run template
    run_parser = subparsers.add_parser("run", help="Run a specific template")
    run_parser.add_argument("template", help="Template name (e.g., ARIMA_Python)")
    run_parser.add_argument("--data", help="Path to data file (overrides config)")
    run_parser.add_argument("--config", help="Path to config file (overrides template config)")
    
    # Validate data
    validate_parser = subparsers.add_parser("validate", help="Validate time series data format")
    validate_parser.add_argument("data", help="Path to data file")
    
    # Recommend templates
    recommend_parser = subparsers.add_parser("recommend", help="Recommend templates based on data")
    recommend_parser.add_argument("data", help="Path to data file")
    
    # Benchmark
    benchmark_parser = subparsers.add_parser("benchmark", help="Run multiple templates and compare")
    benchmark_parser.add_argument("--data", required=True, help="Path to data file")
    benchmark_parser.add_argument("--templates", nargs="+", help="Templates to benchmark (default: curated set)")
    benchmark_parser.add_argument("--output", default="outputs/benchmark", help="Output directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "list":
        list_templates()
    elif args.command == "run":
        run_template(args.template, data_path=args.data, config_path=args.config)
    elif args.command == "validate":
        success = validate_data(args.data)
        sys.exit(0 if success else 1)
    elif args.command == "recommend":
        recommend(args.data)
    elif args.command == "benchmark":
        benchmark(args.data, templates=args.templates, output_dir=args.output)


if __name__ == "__main__":
    main()

