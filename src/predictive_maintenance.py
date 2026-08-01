#!/usr/bin/env python3
"""Predictive maintenance feature engineering utilities."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List


def calculate_rul(
    df: pd.DataFrame,
    asset_id_col: str,
    cycle_col: str,
    rul_col: str = "RUL",
) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each asset.
    
    RUL = max_cycle_for_asset - current_cycle
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with asset IDs and cycle numbers
    asset_id_col : str
        Column name for asset/equipment ID
    cycle_col : str
        Column name for cycle/time step
    rul_col : str
        Name for the new RUL column (default: "RUL")
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with RUL column added
    """
    df = df.copy()
    
    # Get maximum cycle for each asset
    max_cycles = df.groupby(asset_id_col)[cycle_col].max().to_frame(name="max_cycle")
    max_cycles[asset_id_col] = max_cycles.index
    
    # Merge to get max cycle for each row
    df = df.merge(max_cycles, on=asset_id_col, how="left")
    
    # Calculate RUL
    df[rul_col] = df["max_cycle"] - df[cycle_col]
    
    # Drop temporary max_cycle column
    df = df.drop("max_cycle", axis=1)
    
    return df


def create_rul_labels(
    df: pd.DataFrame,
    rul_col: str = "RUL",
    warning_threshold: int = 30,
    critical_threshold: int = 15,
) -> pd.DataFrame:
    """
    Create classification labels based on RUL thresholds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with RUL column
    rul_col : str
        Column name for RUL
    warning_threshold : int
        RUL threshold for warning state (default: 30)
    critical_threshold : int
        RUL threshold for critical state (default: 15)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with label columns added:
        - 'health_status': 'healthy', 'warning', 'critical'
        - 'binary_label': 0 (healthy) or 1 (unhealthy)
        - 'multi_class_label': 0 (healthy), 1 (warning), 2 (critical)
    """
    df = df.copy()
    
    # Binary classification: healthy (0) vs unhealthy (1)
    df["binary_label"] = (df[rul_col] <= warning_threshold).astype(int)
    
    # Multi-class classification
    df["multi_class_label"] = np.where(
        df[rul_col] <= critical_threshold,
        2,  # critical
        np.where(
            df[rul_col] <= warning_threshold,
            1,  # warning
            0,  # healthy
        ),
    )
    
    # Categorical health status
    df["health_status"] = np.where(
        df[rul_col] <= critical_threshold,
        "critical",
        np.where(
            df[rul_col] <= warning_threshold,
            "warning",
            "healthy",
        ),
    )
    
    return df


def add_rolling_statistics(
    df: pd.DataFrame,
    feature_cols: List[str],
    asset_id_col: str,
    window_size: int = 5,
    stats: List[str] = ["mean", "std"],
) -> pd.DataFrame:
    """
    Add rolling window statistics for sensor/feature columns.
    
    Calculates rolling statistics (mean, std, etc.) for each asset separately.
    Useful for capturing degradation trends in predictive maintenance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sensor/feature columns
    feature_cols : List[str]
        List of column names to calculate rolling stats for
    asset_id_col : str
        Column name for asset ID (to group by)
    window_size : int
        Rolling window size (default: 5)
    stats : List[str]
        Statistics to calculate: 'mean', 'std', 'min', 'max', 'median' (default: ['mean', 'std'])
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new rolling statistic columns added
        Column names: {feature}_{stat}_roll{window_size}
    """
    df = df.copy()
    
    # Group by asset to calculate rolling stats per asset
    new_cols = []
    
    for asset_id in df[asset_id_col].unique():
        asset_mask = df[asset_id_col] == asset_id
        asset_data = df[asset_mask].copy()
        
        for col in feature_cols:
            if col not in asset_data.columns:
                continue
            
            for stat in stats:
                if stat == "mean":
                    values = asset_data[col].rolling(window=window_size, min_periods=1).mean()
                elif stat == "std":
                    values = asset_data[col].rolling(window=window_size, min_periods=1).std().fillna(0)
                elif stat == "min":
                    values = asset_data[col].rolling(window=window_size, min_periods=1).min()
                elif stat == "max":
                    values = asset_data[col].rolling(window=window_size, min_periods=1).max()
                elif stat == "median":
                    values = asset_data[col].rolling(window=window_size, min_periods=1).median()
                else:
                    raise ValueError(f"Unknown statistic: {stat}")
                
                new_col_name = f"{col}_{stat}_roll{window_size}"
                asset_data[new_col_name] = values
                new_cols.append(new_col_name)
        
        # Update original dataframe
        df.loc[asset_mask, asset_data.columns] = asset_data.values
    
    return df


def calculate_degradation_rate(
    df: pd.DataFrame,
    feature_cols: List[str],
    asset_id_col: str,
    cycle_col: str,
) -> pd.DataFrame:
    """
    Calculate degradation rate (slope) for sensor features.
    
    Computes the rate of change for each feature over time for each asset.
    Useful for identifying accelerating degradation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sensor columns
    feature_cols : List[str]
        List of column names to calculate degradation rates for
    asset_id_col : str
        Column name for asset ID
    cycle_col : str
        Column name for cycle/time
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with degradation rate columns added
        Column names: {feature}_degradation_rate
    """
    df = df.copy()
    
    from sklearn.linear_model import LinearRegression
    
    for asset_id in df[asset_id_col].unique():
        asset_mask = df[asset_id_col] == asset_id
        asset_data = df[asset_mask].sort_values(cycle_col)
        
        cycles = asset_data[cycle_col].values.reshape(-1, 1)
        
        for col in feature_cols:
            if col not in asset_data.columns:
                continue
            
            values = asset_data[col].values
            
            # Fit linear regression to get slope (degradation rate)
            if len(cycles) > 1:
                lr = LinearRegression()
                lr.fit(cycles, values)
                degradation_rate = lr.coef_[0]
            else:
                degradation_rate = 0.0
            
            # Add as constant column for this asset
            new_col_name = f"{col}_degradation_rate"
            df.loc[asset_mask, new_col_name] = degradation_rate
    
    return df


def prepare_pm_features(
    df: pd.DataFrame,
    asset_id_col: str,
    cycle_col: str,
    feature_cols: List[str],
    calculate_rul_flag: bool = True,
    add_labels: bool = True,
    add_rolling_stats: bool = True,
    add_degradation_rates: bool = False,
    rolling_window: int = 5,
    warning_threshold: int = 30,
    critical_threshold: int = 15,
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline for predictive maintenance.
    
    This function combines all PM feature engineering steps:
    1. Calculate RUL
    2. Create health labels
    3. Add rolling statistics
    4. Add degradation rates (optional)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with asset data
    asset_id_col : str
        Column name for asset ID
    cycle_col : str
        Column name for cycle/time
    feature_cols : List[str]
        List of sensor/feature column names
    calculate_rul_flag : bool
        Whether to calculate RUL (default: True)
    add_labels : bool
        Whether to add health labels (default: True)
    add_rolling_stats : bool
        Whether to add rolling statistics (default: True)
    add_degradation_rates : bool
        Whether to add degradation rates (default: False)
    rolling_window : int
        Window size for rolling statistics (default: 5)
    warning_threshold : int
        RUL threshold for warning (default: 30)
    critical_threshold : int
        RUL threshold for critical (default: 15)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all PM features added
    """
    df = df.copy()
    
    if calculate_rul_flag:
        df = calculate_rul(df, asset_id_col, cycle_col)
    
    if add_labels:
        df = create_rul_labels(
            df,
            rul_col="RUL",
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )
    
    if add_rolling_stats:
        df = add_rolling_statistics(
            df,
            feature_cols=feature_cols,
            asset_id_col=asset_id_col,
            window_size=rolling_window,
        )
    
    if add_degradation_rates:
        df = calculate_degradation_rate(
            df,
            feature_cols=feature_cols,
            asset_id_col=asset_id_col,
            cycle_col=cycle_col,
        )
    
    return df

