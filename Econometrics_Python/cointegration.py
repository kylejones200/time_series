import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint


def adf_test(series: pd.Series, name: str):
    """Perform the Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series)
    return {
        "Test Statistic": result[0],
        "P-Value": result[1],
        "Stationary": result[1] <= 0.05,
    }


def cointegration_test(series1: pd.Series, series2: pd.Series):
    """Perform the Engle-Granger cointegration test."""
    score, p_value, _ = coint(series1, series2)
    return {
        "Cointegration Score": score,
        "P-Value": p_value,
        "Cointegrated": p_value <= 0.05,
    }


def plot_series(data: pd.DataFrame, title: str = "Time Series Data"):
    """Plot multiple time series from a DataFrame."""
    data.plot(subplots=True, figsize=(10, 8), title=title)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()
