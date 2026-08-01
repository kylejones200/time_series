#!/usr/bin/env python3
"""Classical ARIMA baselines using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from collections import deque
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.config import parse_common_config

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    horizon: int
    n_splits: int
    season: int
    max_lag: int
    output_dir: Path
    uni_multi_plot: Path
    baseline_plot: Path
    ensemble_plot: Path
    streaming_plot: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    common = parse_common_config(config_dict, script_dir)

    
    return Config(
        data_path=common.data_path,
        date_col=common.date_col,
        value_col=common.value_col,
        freq=config_dict["data"].get("freq", "MS"),
        horizon=int(config_dict["evaluation"]["horizon"]),
        n_splits=int(config_dict["evaluation"]["n_splits"]),
        season=int(config_dict["evaluation"]["season"]),
        max_lag=int(config_dict["evaluation"]["max_lag"]),
        output_dir=common.output_dir,
        uni_multi_plot=common.output_dir / config_dict["output"]["uni_multi_plot"],
        baseline_plot=common.output_dir / config_dict["output"]["baseline_plot"],
        ensemble_plot=common.output_dir / config_dict["output"]["ensemble_plot"],
        streaming_plot=common.output_dir / config_dict["output"]["streaming_plot"],
    )


def load_series(config: Config) -> pd.Series:
    """Load time series using consolidated loader."""
    from src import load_time_series
    series = load_time_series(
        str(config.data_path),
        date_column=config.date_col,
        value_column=config.value_col
    )
    
    if config.freq:
        series = series.asfreq(config.freq)

    # Avoid leakage: only forward-fill missing values from past observations.
    return series.astype(float).ffill().dropna()


def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create compact seasonal features with stable scale."""
    df = pd.DataFrame(index=index)
    inferred = pd.infer_freq(index)
    is_daily = inferred in {"D", "B", "H"} or (inferred is None and len(index) > 40)
    if is_daily:
        dow = df.index.dayofweek.values
        doy = df.index.dayofyear.values
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    else:
        month = df.index.month.values
        df["sin12"] = np.sin(2 * np.pi * month / 12.0)
        df["cos12"] = np.cos(2 * np.pi * month / 12.0)
    df["trend"] = np.arange(len(df), dtype=float)
    return df


def rolling_origin_uni_vs_multi(
    series: pd.Series, config: Config
) -> Tuple[float, float, pd.Series, pd.Series, pd.Series]:
    """Rolling origin evaluation for univariate vs multivariate models."""
    idx = np.arange(len(series))
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    uni_maes = []
    mul_maes = []
    last_true = None
    last_uni_pred = None
    last_mul_pred = None
    
    for train_idx, _ in splitter.split(idx):
        end_idx = train_idx[-1]
        train_series = series.iloc[: end_idx + 1]
        future_series = series.iloc[end_idx + 1 : end_idx + 1 + config.horizon]
        
        if future_series.empty:
            continue
        
        # Univariate ETS
        uni_model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="add",
            seasonal_periods=config.season,
        ).fit(optimized=True)
        uni_forecast = uni_model.forecast(len(future_series))
        uni_mae = mean_absolute_error(future_series.values, uni_forecast.values)
        uni_maes.append(uni_mae)
        
        # Multivariate regression
        cal_features = make_calendar_features(train_series.index)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(cal_features.values)
        y_train = train_series.values
        
        reg = LinearRegression().fit(X_train, y_train)
        
        future_cal_features = make_calendar_features(future_series.index)
        # Keep feature layout consistent between train and future windows.
        future_cal_features = future_cal_features.reindex(columns=cal_features.columns, fill_value=0.0)
        X_future = scaler.transform(future_cal_features.values)
        mul_forecast = pd.Series(reg.predict(X_future), index=future_series.index)
        mul_mae = mean_absolute_error(future_series.values, mul_forecast.values)
        mul_maes.append(mul_mae)
        
        last_true = future_series
        last_uni_pred = uni_forecast
        last_mul_pred = mul_forecast
    
    mean_uni_mae = float(np.mean(uni_maes)) if uni_maes else float("nan")
    mean_mul_mae = float(np.mean(mul_maes)) if mul_maes else float("nan")
    
    print(f"Univariate (ETS) MAE: {mean_uni_mae:.3f}")
    print(f"Multivariate (Regression) MAE: {mean_mul_mae:.3f}")
    
    return mean_uni_mae, mean_mul_mae, last_true, last_uni_pred, last_mul_pred


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Load series
    series = load_series(config)
    print(f"Loaded {len(series)} data points")
    
    # Rolling origin evaluation
    _, _, last_true, last_uni_pred, last_mul_pred = rolling_origin_uni_vs_multi(series, config)
    
    # Create visualization
    if last_true is not None and last_uni_pred is not None and last_mul_pred is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series.index[-100:], series.values[-100:], "k-", lw=1.5, label="History", alpha=0.8)
        ax.plot(last_true.index, last_true.values, "b-", lw=1.8, label="Actual", alpha=0.8)
        ax.plot(last_uni_pred.index, last_uni_pred.values, "r--", lw=2.0, label="Univariate (ETS)", alpha=0.8)
        ax.plot(last_mul_pred.index, last_mul_pred.values, "g--", lw=2.0, label="Multivariate (Regression)", alpha=0.8)
        
        ax.set_title("Univariate vs Multivariate Forecast Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        save_plot(fig, config.uni_multi_plot, dpi=300)
        plt.close(fig)
        print(f" Plot saved -> {config.uni_multi_plot}")
    
    print("\n ARIMA baseline analysis complete")


if __name__ == "__main__":
    main()
