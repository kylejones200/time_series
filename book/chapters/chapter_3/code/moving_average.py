import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def arithmetic_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate the arithmetic moving average."""
    return series.rolling(window=window).mean()

def geometric_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate the geometric moving average."""
    return series.rolling(window=window).apply(lambda x: np.prod(x)**(1/window), raw=True)

def plot_moving_average(series: pd.Series, ama: pd.Series, gma: pd.Series, title: str = "Moving Averages"):
    """Plot original series with arithmetic and geometric moving averages."""
    plt.plot(series, label='Original Data')
    plt.plot(ama, label='Arithmetic Moving Average', linestyle='--')
    plt.plot(gma, label='Geometric Moving Average', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

