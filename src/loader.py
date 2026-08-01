#!/usr/bin/env python3
"""Load time series data from CSV files."""

import pandas as pd
from pathlib import Path
from typing import Union


def load_time_series(
    file_path: Union[str, Path],
    date_column: str = "date",
    value_column: str = "value",
) -> pd.Series:
    """
    Load time series from CSV file.
    
    Properly handles file closing on Windows to avoid file locking issues.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to CSV file
    date_column : str
        Name of date column (default: "date")
    value_column : str
        Name of value column (default: "value")
        
    Returns:
    --------
    pd.Series
        Time series with datetime index
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # pd.read_csv handles file closing automatically, but using explicit encoding
    # for cross-platform compatibility
    df = pd.read_csv(file_path, encoding="utf-8")
    
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    
    # Drop rows with invalid dates
    df = df.dropna(subset=[date_column])
    
    # Set date as index
    df = df.set_index(date_column)
    
    # Extract value column as Series and sort
    series = df[value_column].sort_index()
    
    # Drop any remaining NaN values
    series = series.dropna()
    
    return series

