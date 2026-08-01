#!/usr/bin/env python3
"""
Arps Decline Curve Models

Implements the Arps decline curve equations for oil & gas production forecasting.

References:
- Arps, J.J. (1945): "Analysis of Decline Curves", Transactions of the AIME
- https://petrowiki.spe.org/Production_forecasting
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Optional, Tuple
import warnings


class ArpsExponential:
    """
    Arps Exponential Decline (b=0)
    
    Equation: q(t) = q_i * exp(-D_i * t)
    
    Where:
    - q_i: initial production rate
    - D_i: initial decline rate
    - t: time from start
    """

    def __init__(self):
        self.q_i: Optional[float] = None
        self.D_i: Optional[float] = None
        self.params: dict = {}

    def fit(self, production: pd.Series) -> "ArpsExponential":
        """
        Fit exponential decline model to production data.
        
        Parameters:
        -----------
        production : pd.Series
            Production rates with datetime index
            
        Returns:
        --------
        self
        """
        if len(production) < 2:
            raise ValueError("Need at least 2 data points to fit model")

        # Ensure non-negative
        production = production.clip(lower=0.001)  # Avoid log(0)

        # Convert time to days from start
        time_days = (production.index - production.index[0]).days
        rates = production.values

        # Initial guess
        q_i_guess = rates[0]
        D_i_guess = -np.log(rates[-1] / rates[0]) / max(time_days[-1], 1) if rates[-1] > 0 else 0.01

        try:
            # Fit exponential decay
            popt, _ = curve_fit(
                lambda t, q_i, D_i: q_i * np.exp(-D_i * t),
                time_days,
                rates,
                p0=[q_i_guess, D_i_guess],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=1000,
            )
            self.q_i, self.D_i = popt
            self.params = {"q_i": self.q_i, "D_i": self.D_i, "b": 0.0}
        except Exception as e:
            warnings.warn(f"Fitting failed: {e}. Using simple estimates.")
            self.q_i = rates[0]
            self.D_i = -np.log(rates[-1] / rates[0]) / max(time_days[-1], 1) if rates[-1] > 0 else 0.01
            self.params = {"q_i": self.q_i, "D_i": self.D_i, "b": 0.0}

        return self

    def predict(
        self, start_date: pd.Timestamp, periods: int, freq: str = "MS"
    ) -> pd.Series:
        """
        Generate forecast using fitted model.
        
        Parameters:
        -----------
        start_date : pd.Timestamp
            Starting date for forecast
        periods : int
            Number of periods to forecast
        freq : str
            Frequency string (e.g., 'MS' for monthly start)
            
        Returns:
        --------
        pd.Series
            Forecasted production rates with datetime index
        """
        if self.q_i is None or self.D_i is None:
            raise ValueError("Model must be fitted before prediction")

        forecast_dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        time_days = (forecast_dates - forecast_dates[0]).days

        forecast = self.q_i * np.exp(-self.D_i * time_days)

        return pd.Series(forecast, index=forecast_dates, name="forecast")


class ArpsHyperbolic:
    """
    Arps Hyperbolic Decline (0 < b < 1)
    
    Equation: q(t) = q_i / (1 + b * D_i * t)^(1/b)
    
    Where:
    - q_i: initial production rate
    - D_i: initial decline rate
    - b: decline exponent (0 < b < 1)
    - t: time from start
    """

    def __init__(self, b: Optional[float] = None):
        self.q_i: Optional[float] = None
        self.D_i: Optional[float] = None
        self.b: Optional[float] = b  # If None, will be fitted
        self.params: dict = {}

    def fit(self, production: pd.Series) -> "ArpsHyperbolic":
        """
        Fit hyperbolic decline model to production data.
        
        Parameters:
        -----------
        production : pd.Series
            Production rates with datetime index
            
        Returns:
        --------
        self
        """
        if len(production) < 3:
            raise ValueError("Need at least 3 data points to fit hyperbolic model")

        production = production.clip(lower=0.001)
        time_days = (production.index - production.index[0]).days
        rates = production.values

        # Initial guesses
        q_i_guess = rates[0]
        D_i_guess = 0.01
        b_guess = self.b if self.b is not None else 0.5

        try:
            if self.b is not None:
                # Fix b, fit q_i and D_i
                popt, _ = curve_fit(
                    lambda t, q_i, D_i: q_i / (1 + self.b * D_i * t) ** (1 / self.b),
                    time_days,
                    rates,
                    p0=[q_i_guess, D_i_guess],
                    bounds=([0, 0], [np.inf, np.inf]),
                    maxfev=1000,
                )
                self.q_i, self.D_i = popt
                self.b = self.b
            else:
                # Fit all three parameters
                popt, _ = curve_fit(
                    lambda t, q_i, D_i, b: q_i / (1 + b * D_i * t) ** (1 / b),
                    time_days,
                    rates,
                    p0=[q_i_guess, D_i_guess, b_guess],
                    bounds=([0, 0, 0.01], [np.inf, np.inf, 0.99]),
                    maxfev=2000,
                )
                self.q_i, self.D_i, self.b = popt

            self.params = {"q_i": self.q_i, "D_i": self.D_i, "b": self.b}
        except Exception as e:
            warnings.warn(f"Hyperbolic fitting failed: {e}. Using exponential approximation.")
            # Fall back to exponential
            self.q_i = rates[0]
            self.D_i = -np.log(rates[-1] / rates[0]) / max(time_days[-1], 1) if rates[-1] > 0 else 0.01
            self.b = 0.0
            self.params = {"q_i": self.q_i, "D_i": self.D_i, "b": 0.0}

        return self

    def predict(
        self, start_date: pd.Timestamp, periods: int, freq: str = "MS"
    ) -> pd.Series:
        """Generate forecast using fitted model."""
        if self.q_i is None or self.D_i is None or self.b is None:
            raise ValueError("Model must be fitted before prediction")

        forecast_dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        time_days = (forecast_dates - forecast_dates[0]).days

        forecast = self.q_i / (1 + self.b * self.D_i * time_days) ** (1 / self.b)

        return pd.Series(forecast, index=forecast_dates, name="forecast")


class ArpsHarmonic:
    """
    Arps Harmonic Decline (b=1)
    
    Equation: q(t) = q_i / (1 + D_i * t)
    
    Special case of hyperbolic decline with b=1.
    """

    def __init__(self):
        self.q_i: Optional[float] = None
        self.D_i: Optional[float] = None
        self.params: dict = {}

    def fit(self, production: pd.Series) -> "ArpsHarmonic":
        """Fit harmonic decline model (b=1 hyperbolic)."""
        hyperbolic = ArpsHyperbolic(b=1.0)
        hyperbolic.fit(production)
        self.q_i = hyperbolic.q_i
        self.D_i = hyperbolic.D_i
        self.params = {"q_i": self.q_i, "D_i": self.D_i, "b": 1.0}
        return self

    def predict(
        self, start_date: pd.Timestamp, periods: int, freq: str = "MS"
    ) -> pd.Series:
        """Generate forecast using fitted model."""
        if self.q_i is None or self.D_i is None:
            raise ValueError("Model must be fitted before prediction")

        forecast_dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        time_days = (forecast_dates - forecast_dates[0]).days

        forecast = self.q_i / (1 + self.D_i * time_days)

        return pd.Series(forecast, index=forecast_dates, name="forecast")

