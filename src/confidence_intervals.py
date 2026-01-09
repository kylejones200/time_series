#!/usr/bin/env python3
"""Bootstrap confidence intervals for time series forecasts."""

import numpy as np
import pandas as pd
from typing import Callable, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


def bootstrap_confidence_intervals(
    model_fit_func: Callable,
    data: pd.Series,
    forecast_steps: int,
    n_bootstraps: int = 100,
    confidence: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate bootstrap confidence intervals for forecasts.
    
    Parameters:
    -----------
    model_fit_func : Callable
        Function that takes data and returns a fitted model with a `forecast(steps)` method
        Example: lambda data: ARIMA(data, order=(1,1,1)).fit()
    data : pd.Series
        Time series data
    forecast_steps : int
        Number of steps to forecast
    n_bootstraps : int
        Number of bootstrap iterations (default: 100)
    confidence : float
        Confidence level (default: 0.95)
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (mean_forecast, lower_bound, upper_bound) as numpy arrays
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    forecasts = []
    successful_bootstraps = 0
    
    for i in range(n_bootstraps):
        try:
            # Bootstrap resample with replacement
            sample = data.sample(n=len(data), replace=True).sort_index()
            
            # Fit model on bootstrap sample
            model = model_fit_func(sample)
            
            # Generate forecast
            if hasattr(model, 'forecast'):
                forecast = model.forecast(steps=forecast_steps)
            elif hasattr(model, 'predict'):
                # For models that use predict instead of forecast
                future_index = pd.date_range(
                    start=data.index[-1] + pd.Timedelta(days=1),
                    periods=forecast_steps,
                    freq=pd.infer_freq(data.index) or 'D'
                )
                forecast = model.predict(future_index)
            else:
                raise AttributeError("Model must have 'forecast' or 'predict' method")
            
            # Convert to numpy array if needed
            if isinstance(forecast, pd.Series):
                forecast = forecast.values
            elif isinstance(forecast, np.ndarray):
                pass
            else:
                forecast = np.array(forecast)
            
            forecasts.append(forecast)
            successful_bootstraps += 1
            
        except (ValueError, AttributeError, RuntimeError) as e:
            # Skip failed bootstrap iterations (model fitting errors, not import errors)
            continue
    
    if successful_bootstraps == 0:
        raise RuntimeError("All bootstrap iterations failed. Check model_fit_func and data.")
    
    if successful_bootstraps < n_bootstraps * 0.5:
        import warnings
        warnings.warn(
            f"Only {successful_bootstraps}/{n_bootstraps} bootstrap iterations succeeded. "
            "Results may be unreliable."
        )
    
    forecasts = np.array(forecasts)
    
    # Calculate percentiles
    alpha = (1 - confidence) / 2
    mean_forecast = np.mean(forecasts, axis=0)
    lower_bound = np.percentile(forecasts, alpha * 100, axis=0)
    upper_bound = np.percentile(forecasts, (1 - alpha) * 100, axis=0)
    
    return mean_forecast, lower_bound, upper_bound


def parametric_confidence_intervals(
    model,
    forecast_steps: int,
    confidence: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate parametric confidence intervals from model.
    
    Parameters:
    -----------
    model
        Fitted model with get_forecast() method (e.g., statsmodels ARIMA)
    forecast_steps : int
        Number of steps to forecast
    confidence : float
        Confidence level (default: 0.95)
    
    Returns:
    --------
    tuple
        (mean_forecast, lower_bound, upper_bound) as numpy arrays
    """
    if not hasattr(model, 'get_forecast'):
        raise AttributeError("Model must have 'get_forecast' method for parametric CIs")
    
    forecast_result = model.get_forecast(steps=forecast_steps)
    mean_forecast = forecast_result.predicted_mean.values
    conf_int = forecast_result.conf_int(alpha=1 - confidence)
    
    lower_bound = conf_int.iloc[:, 0].values
    upper_bound = conf_int.iloc[:, 1].values
    
    return mean_forecast, lower_bound, upper_bound


def compare_ci_methods(
    model_fit_func: Callable,
    data: pd.Series,
    forecast_steps: int,
    n_bootstraps: int = 100,
    confidence: float = 0.95,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compare bootstrap vs parametric confidence intervals.
    
    Parameters:
    -----------
    model_fit_func : Callable
        Function that fits model and returns fitted model
    data : pd.Series
        Time series data
    forecast_steps : int
        Number of steps to forecast
    n_bootstraps : int
        Number of bootstrap iterations
    confidence : float
        Confidence level
    random_seed : int, optional
        Random seed
    
    Returns:
    --------
    pd.DataFrame
        Comparison of CI widths and coverage
    """
    # Bootstrap CIs
    boot_mean, boot_lower, boot_upper = bootstrap_confidence_intervals(
        model_fit_func, data, forecast_steps, n_bootstraps, confidence, random_seed
    )
    boot_width = boot_upper - boot_lower
    
    # Parametric CIs (if available)
    try:
        model = model_fit_func(data)
        if hasattr(model, 'get_forecast'):
            param_mean, param_lower, param_upper = parametric_confidence_intervals(
                model, forecast_steps, confidence
            )
            param_width = param_upper - param_lower
            
            comparison = pd.DataFrame({
                'step': range(1, forecast_steps + 1),
                'bootstrap_mean': boot_mean,
                'bootstrap_lower': boot_lower,
                'bootstrap_upper': boot_upper,
                'bootstrap_width': boot_width,
                'parametric_mean': param_mean,
                'parametric_lower': param_lower,
                'parametric_upper': param_upper,
                'parametric_width': param_width,
                'width_difference': boot_width - param_width,
            })
        else:
            comparison = pd.DataFrame({
                'step': range(1, forecast_steps + 1),
                'bootstrap_mean': boot_mean,
                'bootstrap_lower': boot_lower,
                'bootstrap_upper': boot_upper,
                'bootstrap_width': boot_width,
            })
    except (AttributeError, RuntimeError, ValueError):
        # Parametric CIs not available (model doesn't support it)
        comparison = pd.DataFrame({
            'step': range(1, forecast_steps + 1),
            'bootstrap_mean': boot_mean,
            'bootstrap_lower': boot_lower,
            'bootstrap_upper': boot_upper,
            'bootstrap_width': boot_width,
        })
    
    return comparison

