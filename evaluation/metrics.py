#!/usr/bin/env python3
"""
Evaluation metrics for time series forecasts.

Provides standard metrics for comparing forecasts against actual values.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for evaluation metrics."""

    mae: float
    rmse: float
    mape: float
    r2: float
    mse: float
    mape_valid: bool = True  # False if MAPE calculation failed due to zeros

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape if self.mape_valid else np.nan,
            "R²": self.r2,
            "MSE": self.mse,
        }


def calculate_metrics(
    actual: Union[pd.Series, np.ndarray],
    forecast: Union[pd.Series, np.ndarray],
    name: Optional[str] = None,
) -> MetricResult:
    """
    Calculate evaluation metrics for forecast vs actual.
    
    Parameters:
    -----------
    actual : pd.Series or np.ndarray
        Actual values
    forecast : pd.Series or np.ndarray
        Forecasted values
    name : str, optional
        Name for the forecast (for error messages)
        
    Returns:
    --------
    MetricResult
        Container with all calculated metrics
    """
    # Convert to numpy arrays if needed
    if isinstance(actual, pd.Series):
        actual = actual.values
    if isinstance(forecast, pd.Series):
        forecast = forecast.values

    # Ensure same length
    if len(actual) != len(forecast):
        raise ValueError(
            f"Length mismatch: actual={len(actual)}, forecast={len(forecast)}"
            + (f" (model: {name})" if name else "")
        )

    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    if mask.sum() == 0:
        raise ValueError("No valid data points after removing NaN")

    actual_clean = actual[mask]
    forecast_clean = forecast[mask]

    # Calculate metrics
    mse = np.mean((actual_clean - forecast_clean) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_clean - forecast_clean))

    # R²
    ss_res = np.sum((actual_clean - forecast_clean) ** 2)
    ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE (handle zeros in actual)
    non_zero_mask = actual_clean != 0
    mape_valid = non_zero_mask.sum() > 0
    if mape_valid:
        mape = np.mean(np.abs((actual_clean[non_zero_mask] - forecast_clean[non_zero_mask]) / actual_clean[non_zero_mask])) * 100
    else:
        mape = np.nan

    return MetricResult(
        mae=mae,
        rmse=rmse,
        mape=mape,
        r2=r2,
        mse=mse,
        mape_valid=mape_valid,
    )

