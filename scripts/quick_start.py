#!/usr/bin/env python3
"""
Quick Start Wizard

Interactive wizard to help new users get started with forecasting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src import load_time_series


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{default_str}]: ").strip().lower()
    
    if not response:
        return default
    return response in ["y", "yes"]


def ask_choice(prompt: str, choices: list, default: int = 0) -> str:
    """Ask user to choose from a list."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = "→" if i == default + 1 else " "
        print(f"  {marker} [{i}] {choice}")
    
    while True:
        try:
            response = input(f"\nEnter choice [1-{len(choices)}] (default: {default + 1}): ").strip()
            if not response:
                return choices[default]
            
            idx = int(response) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print("Please enter a valid number")


def quick_start_wizard():
    """Run the quick start wizard."""
    print_header(" Time Series Forecasting - Quick Start Wizard")
    
    print("This wizard will help you:")
    print("  1. Validate your data")
    print("  2. Recommend suitable templates")
    print("  3. Generate your first forecast")
    print()
    
    if not ask_yes_no("Ready to start?", default=True):
        print("Exiting. Run this wizard anytime with: python scripts/quick_start.py")
        return
    
    # Step 1: Data location
    print_header("Step 1: Locate Your Data")
    
    data_path = input("Enter path to your CSV file: ").strip()
    data_path = Path(data_path).expanduser()
    
    if not data_path.exists():
        print(f" File not found: {data_path}")
        if ask_yes_no("Generate example data instead?", default=True):
            print("\nGenerating example data...")
            from data.production.generate_example_data import main as generate_data
            generate_data()
            data_path = Path("data/production/well_production.csv")
            print(f" Example data created at: {data_path}")
        else:
            return
    
    # Step 2: Validate data
    print_header("Step 2: Validate Data")
    
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
        print(f" File loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check columns
        if "date" not in df.columns or "value" not in df.columns:
            print("️  Expected columns: 'date' and 'value'")
            print(f"   Found columns: {list(df.columns)}")
            
            if ask_yes_no("Would you like to map your columns?", default=True):
                date_col = ask_choice("Select date column:", list(df.columns))
                value_col = ask_choice("Select value column:", list(df.columns))
                
                # Create a mapped version
                df_mapped = df.rename(columns={date_col: "date", value_col: "value"})
                mapped_path = data_path.parent / f"{data_path.stem}_mapped.csv"
                df_mapped[["date", "value"]].to_csv(mapped_path, index=False)
                data_path = mapped_path
                print(f" Mapped data saved to: {mapped_path}")
        
        # Load and analyze
        series = load_time_series(str(data_path), date_column="date", value_column="value")
        print(f" Data validated successfully")
        print(f"   Date range: {series.index.min()} to {series.index.max()}")
        print(f"   Value range: {series.min():.2f} to {series.max():.2f}")
        print(f"   Data points: {len(series)}")
        
    except Exception as e:
        print(f" Error validating data: {e}")
        return
    
    # Step 3: Recommend templates
    print_header("Step 3: Template Recommendations")
    
    n_points = len(series)
    
    if n_points < 30:
        recommended = ["MovingAverage_Python", "ExponentialSmoothing_Python"]
        reason = "Short series - simple methods work best"
    elif n_points < 100:
        recommended = ["ARIMA_Python", "Prophet_Python", "ExponentialSmoothing_Python"]
        reason = "Medium series - classical methods recommended"
    else:
        recommended = ["ARIMA_Python", "Prophet_Python", "StatsForecast_Python"]
        reason = "Long series - multiple methods available"
    
    print(f"Based on your data ({n_points} points), we recommend:")
    for i, template in enumerate(recommended, 1):
        print(f"  {i}. {template}")
    print(f"\nReason: {reason}")
    
    # Step 4: Choose action
    print_header("Step 4: Choose Your Action")
    
    actions = [
        "Run a single recommended template",
        "Benchmark multiple templates",
        "Get more recommendations",
        "Exit"
    ]
    
    action = ask_choice("What would you like to do?", actions, default=0)
    
    if "single" in action.lower():
        template = ask_choice("Select template to run:", recommended, default=0)
        print(f"\n Running {template}...")
        print(f"   Command: python forecast.py run {template} --data {data_path}")
        
        from forecast import run_template
        run_template(template, data_path=str(data_path))
        
    elif "benchmark" in action.lower():
        print(f"\n Running benchmark...")
        print(f"   This will compare multiple templates")
        from scripts.auto_benchmark import benchmark
        benchmark(str(data_path), categories=["fast"])
        
    elif "recommendations" in action.lower():
        from forecast import recommend
        recommend(str(data_path))
    
    print_header(" Quick Start Complete!")
    print("Next steps:")
    print("  • Review results in outputs/ directory")
    print("  • Try other templates: python forecast.py list")
    print("  • Read the documentation: docs/sphinx/")
    print("  • Run benchmarks: python forecast.py benchmark --data <your_data.csv>")


if __name__ == "__main__":
    quick_start_wizard()

