#!/usr/bin/env python3
"""
Data validation tool.

Validates time series data format and provides helpful error messages.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_time_series


def validate_time_series_data(data_path: str, date_col: str = "date", value_col: str = "value") -> dict:
    """
    Comprehensive data validation.
    
    Returns:
    --------
    dict
        Validation results with detailed information
    """
    results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "info": {},
        "suggestions": []
    }
    
    data_path = Path(data_path)
    
    # Check file exists
    if not data_path.exists():
        results["errors"].append(f"File not found: {data_path}")
        return results
    
    # Try to load CSV
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
        results["info"]["rows"] = len(df)
        results["info"]["columns"] = list(df.columns)
    except Exception as e:
        results["errors"].append(f"Failed to read CSV: {e}")
        return results
    
    # Check required columns
    if date_col not in df.columns:
        results["errors"].append(f"Missing date column: '{date_col}'")
        results["suggestions"].append(f"Available columns: {list(df.columns)}")
        if len(df.columns) >= 1:
            results["suggestions"].append(f"Did you mean: '{df.columns[0]}'?")
    else:
        # Validate date column
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            invalid_dates = dates.isna().sum()
            if invalid_dates > 0:
                results["warnings"].append(f"{invalid_dates} invalid dates found")
            else:
                results["info"]["date_range"] = f"{dates.min()} to {dates.max()}"
                results["info"]["date_count"] = len(dates.unique())
        except Exception as e:
            results["errors"].append(f"Date column validation failed: {e}")
    
    if value_col not in df.columns:
        results["errors"].append(f"Missing value column: '{value_col}'")
        results["suggestions"].append(f"Available columns: {list(df.columns)}")
    else:
        # Validate value column
        try:
            values = pd.to_numeric(df[value_col], errors="coerce")
            invalid_values = values.isna().sum()
            if invalid_values > 0:
                results["warnings"].append(f"{invalid_values} invalid/missing values found")
            
            results["info"]["value_range"] = f"{values.min():.2f} to {values.max():.2f}"
            results["info"]["value_mean"] = f"{values.mean():.2f}"
            
            # Check for zeros (might indicate issues)
            zero_count = (values == 0).sum()
            if zero_count > len(values) * 0.1:
                results["warnings"].append(f"Many zero values ({zero_count}), may indicate data issues")
            
            # Check for negative values (if not expected)
            negative_count = (values < 0).sum()
            if negative_count > 0:
                results["warnings"].append(f"{negative_count} negative values found")
                
        except Exception as e:
            results["errors"].append(f"Value column validation failed: {e}")
    
    # Check data length
    if len(df) < 20:
        results["warnings"].append(f"Short series ({len(df)} points). Most methods need 20+ points")
        results["suggestions"].append("Consider: MovingAverage_Python or ExponentialSmoothing_Python")
    elif len(df) < 50:
        results["info"]["recommendation"] = "Medium series - good for most methods"
    else:
        results["info"]["recommendation"] = "Long series - all methods available"
    
    # Check for duplicates
    if date_col in df.columns:
        duplicates = df[date_col].duplicated().sum()
        if duplicates > 0:
            results["warnings"].append(f"{duplicates} duplicate dates found")
            results["suggestions"].append("Consider aggregating duplicate dates (mean, sum, etc.)")
    
    # Overall validation
    results["valid"] = len(results["errors"]) == 0
    
    return results


def print_validation_report(results: dict):
    """Print a formatted validation report."""
    print("\n" + "=" * 70)
    print("  Data Validation Report")
    print("=" * 70 + "\n")
    
    if results["valid"]:
        print(" Data is valid and ready to use!\n")
    else:
        print(" Data validation failed\n")
    
    # Errors
    if results["errors"]:
        print("Errors:")
        for error in results["errors"]:
            print(f"   {error}")
        print()
    
    # Warnings
    if results["warnings"]:
        print("Warnings:")
        for warning in results["warnings"]:
            print(f"  ️  {warning}")
        print()
    
    # Info
    if results["info"]:
        print("Information:")
        for key, value in results["info"].items():
            print(f"  ℹ️  {key.replace('_', ' ').title()}: {value}")
        print()
    
    # Suggestions
    if results["suggestions"]:
        print("Suggestions:")
        for suggestion in results["suggestions"]:
            print(f"   {suggestion}")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate time series data format")
    parser.add_argument("data", help="Path to data file")
    parser.add_argument("--date-col", default="date", help="Date column name")
    parser.add_argument("--value-col", default="value", help="Value column name")
    
    args = parser.parse_args()
    
    results = validate_time_series_data(args.data, args.date_col, args.value_col)
    print_validation_report(results)
    
    sys.exit(0 if results["valid"] else 1)

