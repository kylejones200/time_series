#!/usr/bin/env python3
"""Evaluation utilities for time series forecasts."""

import pandas as pd
import numpy as np
from typing import Tuple


class Evaluator:
    """
    Evaluator for time series forecasts.
    
    Holds out the last segment of data and calculates error metrics.
    """
    
    def __init__(self, test_size: float = 0.2):
        """
        Initialize evaluator.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to hold out for testing (default: 0.2)
        """
        self.test_size = test_size
        self.train_data: pd.Series = None
        self.test_data: pd.Series = None
        
    def split(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Split time series into train and test sets.
        
        Parameters:
        -----------
        series : pd.Series
            Full time series
            
        Returns:
        --------
        train : pd.Series
            Training data
        test : pd.Series
            Test data (last segment)
        """
        split_idx = int(len(series) * (1 - self.test_size))
        
        self.train_data = series.iloc[:split_idx]
        self.test_data = series.iloc[split_idx:]
        
        return self.train_data, self.test_data
    
    def evaluate(self, forecast: pd.Series, actual: pd.Series) -> dict:
        """
        Evaluate forecast against actual values.
        
        Parameters:
        -----------
        forecast : pd.Series
            Forecasted values
        actual : pd.Series
            Actual values
            
        Returns:
        --------
        dict
            Dictionary with error metrics
        """
        # Align indices
        common_idx = forecast.index.intersection(actual.index)
        
        if len(common_idx) == 0:
            raise ValueError("No overlapping indices between forecast and actual")
        
        forecast_aligned = forecast.loc[common_idx]
        actual_aligned = actual.loc[common_idx]
        
        # Remove NaN values
        mask = ~(forecast_aligned.isna() | actual_aligned.isna())
        forecast_clean = forecast_aligned[mask]
        actual_clean = actual_aligned[mask]
        
        if len(forecast_clean) == 0:
            raise ValueError("No valid data points after removing NaN")
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((actual_clean - forecast_clean) ** 2))
        
        return {
            "RMSE": rmse,
            "n_points": len(forecast_clean),
        }

