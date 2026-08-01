#!/usr/bin/env python3
"""Model wrappers for time series forecasting."""

import pandas as pd
import numpy as np
from typing import Optional
from pmdarima import auto_arima


class ARIMAModel:
    """
    ARIMA model wrapper.
    
    Fits ARIMA model using auto_arima and provides forecast method.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize ARIMA model.
        
        Parameters:
        -----------
        **kwargs
            Arguments passed to pmdarima.auto_arima
        """
        self.model = None
        self.kwargs = kwargs
        
    def fit(self, series: pd.Series) -> "ARIMAModel":
        """
        Fit ARIMA model to time series.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data with datetime index
            
        Returns:
        --------
        self
        """
        # Store training index for forecast date generation
        self.train_index = series.index
        
        # Default auto_arima parameters
        default_kwargs = {
            "start_p": 0,
            "start_q": 0,
            "max_p": 5,
            "max_q": 5,
            "seasonal": False,
            "stepwise": True,
            "suppress_warnings": True,
            "error_action": "ignore",
        }
        
        # Merge with user-provided kwargs
        fit_kwargs = {**default_kwargs, **self.kwargs}
        
        # Fit model (use values only, not index)
        self.model = auto_arima(series.values, **fit_kwargs)
        
        return self
    
    def forecast(self, n_periods: int, return_conf_int: bool = False):
        """
        Generate forecast.
        
        Parameters:
        -----------
        n_periods : int
            Number of periods to forecast
        return_conf_int : bool
            Whether to return confidence intervals
            
        Returns:
        --------
        forecast : pd.Series
            Forecasted values with datetime index
        conf_int : pd.DataFrame, optional
            Confidence intervals if return_conf_int=True
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.train_index is None:
            raise ValueError("Training index not found. Model may not be fitted properly.")
        
        # Generate forecast
        forecast, conf_int = self.model.predict(
            n_periods=n_periods,
            return_conf_int=True,
            alpha=0.05  # 95% confidence interval
        )
        
        # Create forecast index from training index
        last_date = pd.Timestamp(self.train_index[-1])
        
        # Infer frequency from training data
        freq = pd.infer_freq(self.train_index)
        if freq is None:
            # If can't infer, estimate from spacing
            if len(self.train_index) > 1:
                avg_delta = self.train_index[-1] - self.train_index[-2]  # Use last two points
                # Convert to pandas frequency string if possible
                if avg_delta.days == 1:
                    freq = 'D'
                elif avg_delta.days == 7:
                    freq = 'W'
                elif 28 <= avg_delta.days <= 31:
                    freq = 'MS'  # Month start
                elif avg_delta.days == 30 or avg_delta.days == 31:
                    freq = 'MS'
                else:
                    freq = avg_delta  # Use timedelta
            else:
                freq = 'D'  # Default to daily
        
        # Create forecast index
        if isinstance(freq, pd.Timedelta):
            # If freq is a timedelta, calculate next date manually
            next_date = last_date + freq
            forecast_index = pd.date_range(start=next_date, periods=n_periods, freq=freq)
        else:
            # Use frequency string
            next_date = last_date + pd.tseries.frequencies.to_offset(freq)
            forecast_index = pd.date_range(start=next_date, periods=n_periods, freq=freq)
        
        forecast_series = pd.Series(forecast, index=forecast_index, name="forecast")
        
        if return_conf_int:
            conf_int_df = pd.DataFrame(
                conf_int,
                index=forecast_index,
                columns=["lower", "upper"]
            )
            return forecast_series, conf_int_df
        else:
            return forecast_series
    
    def get_order(self) -> tuple:
        """Get ARIMA order (p, d, q)."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.order

