"""
Time Series Utilities
Common utilities for time series data handling, date operations, and preprocessing.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional, List


def load_ts_data(
    file_path: Union[str, Path],
    date_col: str = "date",
    value_col: str = "value",
    index_col: Optional[str] = None,
) -> pd.Series:
    """
    Load time series data from CSV file.

    Parameters:
    -----------
    file_path : str or Path
        Path to CSV file
    date_col : str
        Name of date column
    value_col : str
        Name of value column
    index_col : str, optional
        Column to use as index (if None, uses date_col)

    Returns:
    --------
    pd.Series
        Time series with datetime index
    """
    df = pd.read_csv(file_path)

    df[date_col] = pd.to_datetime(df[date_col])

    index_col = index_col or date_col
    df.set_index(index_col, inplace=True)

    return df[value_col]


def ensure_datetime_index(
    data: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """Ensure data has datetime index."""
    data_map = {
        True: lambda d: (
            d
            if isinstance(d.index, pd.DatetimeIndex)
            else d.set_index(pd.to_datetime(d.index))
        ),
        False: lambda d: d,
    }
    return data_map[not isinstance(data.index, pd.DatetimeIndex)](data)


def resample_ts(data: pd.Series, freq: str = "D", method: str = "mean") -> pd.Series:
    """
    Resample time series to different frequency.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    freq : str
        Target frequency (e.g., 'D', 'W', 'M', 'H')
    method : str
        Aggregation method ('mean', 'sum', 'last', 'first')

    Returns:
    --------
    pd.Series
        Resampled time series
    """
    data = ensure_datetime_index(data)

    method_map = {
        "mean": lambda d: d.resample(freq).mean(),
        "sum": lambda d: d.resample(freq).sum(),
        "last": lambda d: d.resample(freq).last(),
        "first": lambda d: d.resample(freq).first(),
    }

    return method_map.get(method, method_map["mean"])(data)


def create_lags(data: pd.Series, lags: List[int] = [1, 2, 3, 7, 14]) -> pd.DataFrame:
    """
    Create lagged features from time series.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    lags : list of int
        Lag periods to create

    Returns:
    --------
    pd.DataFrame
        DataFrame with original and lagged features
    """
    df = pd.DataFrame({"value": data})

    [df.__setitem__(f"lag_{lag}", data.shift(lag)) for lag in lags]

    return df


def create_rolling_features(
    data: pd.Series,
    windows: List[int] = [7, 14, 30],
    functions: List[str] = ["mean", "std"],
) -> pd.DataFrame:
    """
    Create rolling window features.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    windows : list of int
        Window sizes
    functions : list of str
        Functions to apply ('mean', 'std', 'min', 'max')

    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling features
    """
    df = pd.DataFrame({"value": data})

    function_map = {
        "mean": lambda d, w: d.rolling(w).mean(),
        "std": lambda d, w: d.rolling(w).std(),
        "min": lambda d, w: d.rolling(w).min(),
        "max": lambda d, w: d.rolling(w).max(),
    }

    [
        [
            df.__setitem__(f"rolling_{func}_{window}", function_map[func](data, window))
            for func in functions
        ]
        for window in windows
    ]

    return df


def split_ts(
    data: Union[pd.Series, pd.DataFrame],
    train_size: Union[float, int, None] = None,
    test_size: Union[float, int, None] = None,
    date_split: Optional[str] = None,
) -> tuple:
    """
    Split time series into train and test sets.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data
    train_size : float or int, optional
        If float: proportion of data for training
        If int: number of observations for training
        Default: 0.8 (if test_size not provided)
    test_size : float or int, optional
        If float: proportion of data for testing
        If int: number of observations for testing
        If provided, takes precedence over train_size
    date_split : str, optional
        Date string to split on (e.g., '2023-01-01')

    Returns:
    --------
    tuple
        (train_data, test_data)
    """
    data = ensure_datetime_index(data)

    # Handle date_split first (takes precedence)
    if date_split is not None:
        return (data[:date_split], data[date_split:])

    # Convert test_size to train_size if provided
    if test_size is not None:
        if isinstance(test_size, float):
            train_size = 1.0 - test_size
        else:  # int
            train_size = len(data) - test_size
    elif train_size is None:
        train_size = 0.8  # Default

    # Perform split
    if isinstance(train_size, float):
        split_idx = int(len(data) * train_size)
        return (data[:split_idx], data[split_idx:])
    else:  # int
        return (data[:train_size], data[train_size:])


def detect_frequency(data: pd.Series) -> str:
    """
    Detect the frequency of a time series.

    Parameters:
    -----------
    data : pd.Series
        Time series data

    Returns:
    --------
    str
        Detected frequency string
    """
    data = ensure_datetime_index(data)

    inferred = pd.infer_freq(data.index)

    freq_map = {
        "H": "H",
        "D": "D",
        "W": "W",
        "M": "M",
        "Q": "Q",
        "Y": "Y",
    }

    return freq_map.get(inferred, "D") if inferred else "D"


def fill_missing_dates(data: pd.Series, method: str = "forward") -> pd.Series:
    """
    Fill missing dates in time series.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    method : str
        Fill method ('forward', 'backward', 'interpolate')

    Returns:
    --------
    pd.Series
        Time series with filled missing dates
    """
    data = ensure_datetime_index(data)

    full_index = pd.date_range(
        start=data.index.min(), end=data.index.max(), freq=detect_frequency(data)
    )

    data = data.reindex(full_index)

    method_map = {
        "forward": lambda d: d.fillna(method="ffill"),
        "backward": lambda d: d.fillna(method="bfill"),
        "interpolate": lambda d: d.interpolate(),
    }

    return method_map.get(method, method_map["forward"])(data)


def remove_outliers_iqr(data: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Remove outliers using IQR method.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    factor : float
        IQR factor for outlier detection

    Returns:
    --------
    pd.Series
        Time series with outliers removed
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    return data[(data >= lower_bound) & (data <= upper_bound)]


def create_time_features(data: pd.Series) -> pd.DataFrame:
    """
    Create time-based features from datetime index.

    Parameters:
    -----------
    data : pd.Series
        Time series data with datetime index

    Returns:
    --------
    pd.DataFrame
        DataFrame with time features
    """
    data = ensure_datetime_index(data)

    df = pd.DataFrame({"value": data})

    df["year"] = data.index.year
    df["month"] = data.index.month
    df["day"] = data.index.day
    df["dayofweek"] = data.index.dayofweek
    df["dayofyear"] = data.index.dayofyear
    df["week"] = data.index.isocalendar().week
    df["quarter"] = data.index.quarter

    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = data.index.is_month_start.astype(int)
    df["is_month_end"] = data.index.is_month_end.astype(int)

    return df


def locate_features_with_too_many_missing_values(
    df: pd.DataFrame, threshold: float = 0.1
) -> list[str]:
    """Return columns whose missing-value ratio exceeds the threshold."""
    bad_cols = []
    num_rows = len(df)
    if num_rows == 0:
        return bad_cols
    for col in df.columns:
        nan_perc = df[col].isna().sum() / num_rows
        if nan_perc > threshold:
            bad_cols.append(col)
    return bad_cols


def clean_up_turbine_data(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Clean wind-turbine style dataset: timestamps, duplicates, resample, drop sparse features."""  # noqa: E501
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(
        df["Date_time"], infer_datetime_format=True, utc=True
    )
    df = df.drop_duplicates(subset=["Timestamp"], keep="first")
    df = df.set_index("Timestamp")
    df = df.drop(
        columns=[c for c in ["Wind_turbine_name", "Date_time"] if c in df.columns]
    )
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df = df.sort_index()

    df = df.resample("10T").ffill(limit=1)

    bad_cols = locate_features_with_too_many_missing_values(df, threshold)
    df = df.drop(columns=bad_cols)

    df.index = pd.to_datetime(df.index)
    return df


def map_features(
    df_turbine: pd.DataFrame, df_description: pd.DataFrame
) -> dict[str, list[str]]:
    feature_map: dict[str, list[str]] = {}
    for var_name in df_description["Variable_name"]:
        cols = [c for c in df_turbine.columns if var_name in c]
        feature_map[var_name] = cols
    return feature_map


def order_columns_by_description(
    df_turbine: pd.DataFrame, df_description: pd.DataFrame
) -> pd.DataFrame:
    feature_map = map_features(df_turbine, df_description)
    ordered_frames = [df_turbine[cols] for cols in feature_map.values() if cols]
    if ordered_frames:
        return pd.concat(ordered_frames, axis=1)
    return df_turbine
