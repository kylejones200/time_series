#!/usr/bin/env python3
"""Time series feature engineering utilities."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union


def extract_time_features(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
    index_is_date: bool = True,
) -> pd.DataFrame:
    """
    Extract time-based features from datetime index or column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index or column
    date_column : str, optional
        Name of date column (if index is not datetime)
    index_is_date : bool
        Whether the index is datetime (default: True)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with time features added:
        - hour, day_of_week, day_of_month, day_of_year
        - month, quarter, week_of_year
        - is_weekend, is_month_start, is_month_end
    """
    df = df.copy()
    
    if index_is_date:
        dates = df.index
    else:
        if date_column is None:
            raise ValueError("Must specify date_column if index is not datetime")
        dates = pd.to_datetime(df[date_column])
    
    # Basic time features
    df["hour"] = dates.hour
    df["day_of_week"] = dates.dayofweek  # 0=Monday, 6=Sunday
    df["day_of_month"] = dates.day
    df["day_of_year"] = dates.dayofyear
    df["month"] = dates.month
    df["quarter"] = dates.quarter
    df["week_of_year"] = dates.isocalendar().week
    df["year"] = dates.year
    
    # Boolean features
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = dates.is_month_start.astype(int)
    df["is_month_end"] = dates.is_month_end.astype(int)
    
    return df


def create_lag_features(
    series: pd.Series,
    lags: List[int],
    name_prefix: str = "lag",
) -> pd.DataFrame:
    """
    Create lagged features from time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series
    lags : List[int]
        List of lag periods (e.g., [1, 2, 3, 7])
    name_prefix : str
        Prefix for lag column names (default: "lag")
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features
    """
    df = pd.DataFrame({"original": series})
    
    for lag in lags:
        df[f"{name_prefix}_{lag}"] = series.shift(lag)
    
    return df


def create_rolling_features(
    series: pd.Series,
    windows: List[int],
    functions: List[str] = ["mean", "std"],
    name_prefix: str = "rolling",
) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    Parameters:
    -----------
    series : pd.Series
        Time series
    windows : List[int]
        Window sizes (e.g., [3, 7, 30])
    functions : List[str]
        Functions to apply: "mean", "std", "min", "max", "median" (default: ["mean", "std"])
    name_prefix : str
        Prefix for feature names (default: "rolling")
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling features
    """
    df = pd.DataFrame({"original": series})
    
    for window in windows:
        rolling = series.rolling(window=window, min_periods=1)
        
        for func in functions:
            if func == "mean":
                df[f"{name_prefix}_{func}_{window}"] = rolling.mean()
            elif func == "std":
                df[f"{name_prefix}_{func}_{window}"] = rolling.std().fillna(0)
            elif func == "min":
                df[f"{name_prefix}_{func}_{window}"] = rolling.min()
            elif func == "max":
                df[f"{name_prefix}_{func}_{window}"] = rolling.max()
            elif func == "median":
                df[f"{name_prefix}_{func}_{window}"] = rolling.median()
            else:
                raise ValueError(f"Unknown function: {func}")
    
    return df


def create_differenced_features(
    series: pd.Series,
    orders: List[int] = [1],
    name_prefix: str = "diff",
) -> pd.DataFrame:
    """
    Create differenced features.
    
    Parameters:
    -----------
    series : pd.Series
        Time series
    orders : List[int]
        Differencing orders (default: [1])
    name_prefix : str
        Prefix for feature names (default: "diff")
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with differenced features
    """
    df = pd.DataFrame({"original": series})
    
    for order in orders:
        df[f"{name_prefix}_{order}"] = series.diff(order)
    
    return df


def create_seasonal_features(
    series: pd.Series,
    seasonal_periods: List[int],
    name_prefix: str = "seasonal",
) -> pd.DataFrame:
    """
    Create seasonal features (sine/cosine transformations).
    
    Parameters:
    -----------
    series : pd.Series
        Time series
    seasonal_periods : List[int]
        Seasonal periods (e.g., [12, 365] for monthly/yearly)
    name_prefix : str
        Prefix for feature names (default: "seasonal")
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with seasonal features
    """
    df = pd.DataFrame({"original": series})
    n = len(series)
    
    for period in seasonal_periods:
        t = np.arange(n)
        df[f"{name_prefix}_sin_{period}"] = np.sin(2 * np.pi * t / period)
        df[f"{name_prefix}_cos_{period}"] = np.cos(2 * np.pi * t / period)
    
    return df


def prepare_features(
    series: pd.Series,
    include_time_features: bool = True,
    include_lags: bool = True,
    include_rolling: bool = True,
    include_differenced: bool = False,
    include_seasonal: bool = False,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    seasonal_periods: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with datetime index
    include_time_features : bool
        Extract time-based features (default: True)
    include_lags : bool
        Create lag features (default: True)
    include_rolling : bool
        Create rolling statistics (default: True)
    include_differenced : bool
        Create differenced features (default: False)
    include_seasonal : bool
        Create seasonal features (default: False)
    lags : List[int], optional
        Lag periods (default: [1, 2, 3, 7])
    rolling_windows : List[int], optional
        Rolling window sizes (default: [3, 7, 30])
    seasonal_periods : List[int], optional
        Seasonal periods (default: [12, 365])
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all engineered features
    """
    df = pd.DataFrame({"target": series})
    
    if include_time_features:
        df = extract_time_features(df, index_is_date=True)
    
    if include_lags:
        if lags is None:
            lags = [1, 2, 3, 7]
        lag_df = create_lag_features(series, lags=lags)
        df = pd.concat([df, lag_df.drop("original", axis=1)], axis=1)
    
    if include_rolling:
        if rolling_windows is None:
            rolling_windows = [3, 7, 30]
        rolling_df = create_rolling_features(series, windows=rolling_windows)
        df = pd.concat([df, rolling_df.drop("original", axis=1)], axis=1)
    
    if include_differenced:
        diff_df = create_differenced_features(series)
        df = pd.concat([df, diff_df.drop("original", axis=1)], axis=1)
    
    if include_seasonal:
        if seasonal_periods is None:
            # Infer from frequency
            freq = pd.infer_freq(series.index)
            if freq == "D":
                seasonal_periods = [7, 365]
            elif freq in ["MS", "M"]:
                seasonal_periods = [12]
            elif freq == "H":
                seasonal_periods = [24, 168]  # daily, weekly
            else:
                seasonal_periods = [12]
        
        seasonal_df = create_seasonal_features(series, seasonal_periods=seasonal_periods)
        df = pd.concat([df, seasonal_df.drop("original", axis=1)], axis=1)
    
    return df.dropna()

