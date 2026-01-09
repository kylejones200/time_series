#!/usr/bin/env python3
"""Exponential smoothing evaluations using consolidated utilities."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    save_plot,
    ensure_output_dir,
    get_output_dir,
)

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    history_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    horizon: int
    n_splits: int
    season: int
    output_dir: Path
    ets_plot: Path
    comparison_plot: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    repo_root = script_dir.parent
    data_path = repo_root / "data" / config_dict["data"]["input_file"]
    output_dir = ensure_output_dir(Path(script_dir) / config_dict["output"]["output_dir"])
    
    experiment = config_dict["experiment"]
    
    return Config(
        data_path=data_path,
        date_col=config_dict["data"]["date_col"],
        value_col=config_dict["data"]["value_col"],
        freq=config_dict["data"].get("freq", "MS"),
        history_end=pd.Timestamp(experiment["history_end"]),
        forecast_start=pd.Timestamp(experiment["forecast_start"]),
        forecast_end=pd.Timestamp(experiment["forecast_end"]),
        horizon=int(config_dict["evaluation"]["horizon"]),
        n_splits=int(config_dict["evaluation"]["n_splits"]),
        season=int(config_dict["evaluation"]["season"]),
        output_dir=output_dir,
        ets_plot=output_dir / config_dict["output"]["ets_plot"],
        comparison_plot=output_dir / config_dict["output"]["comparison_plot"],
    )


def load_series(config: Config) -> pd.Series:
    """Load time series using consolidated loader."""
    # Use consolidated loader, then apply any template-specific processing
    series = load_time_series(
        str(config.data_path),
        date_column=config.date_col,
        value_column=config.value_col
    )
    
    # Apply frequency conversion if needed
    if config.freq:
        series = series.asfreq(config.freq)
    
    return series.astype(float)


def rolling_origin_ets(
    series: pd.Series, config: Config
) -> Tuple[float, Optional[pd.Series], Optional[pd.Series]]:
    """Rolling origin evaluation for ETS model."""
    idx = np.arange(len(series))
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    maes = []
    last_true = None
    last_pred = None

    for train_idx, _ in splitter.split(idx):
        end_idx = train_idx[-1]
        train_series = series.iloc[: end_idx + 1]
        future_series = series.iloc[end_idx + 1 : end_idx + 1 + config.horizon]
        
        if future_series.empty:
            continue

        model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="add",
            seasonal_periods=config.season,
        ).fit(optimized=True)
        
        forecast = model.forecast(len(future_series))
        mae = mean_absolute_error(future_series.values, forecast.values)
        maes.append(mae)

        last_true = future_series
        last_pred = forecast

    mean_mae = float(np.mean(maes)) if maes else float("nan")
    print(f"ETS rolling-origin MAE: {mean_mae:.3f}")
    return mean_mae, last_true, last_pred


def plot_ets_forecast(series: pd.Series, config: Config, last_forecast: Optional[pd.Series]) -> None:
    """Plot ETS forecast with confidence intervals."""
    start_2024 = pd.Timestamp("2024-01-01")
    history_end = pd.Timestamp("2024-12-01")
    forecast_index = pd.period_range(
        config.forecast_start, config.forecast_end, freq="M"
    ).to_timestamp()

    history = series.loc[start_2024:history_end]
    actual = series.loc[config.forecast_start : config.forecast_end]

    ets_model = ExponentialSmoothing(
        series.loc[:history_end],
        trend="add",
        seasonal="add",
        seasonal_periods=config.season,
    ).fit(optimized=True)
    
    forecast = ets_model.forecast(len(forecast_index))
    residuals = series.loc[:history_end] - ets_model.fittedvalues
    sigma = float(residuals.std(ddof=1)) if len(residuals) else 0.0
    upper = forecast + 1.96 * sigma
    lower = forecast - 1.96 * sigma

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, color="#555555", lw=1.5, label="History")
    ax.axvline(config.forecast_start, color="#777777", linestyle="--", lw=1)
    
    if not actual.empty:
        ax.plot(actual.index, actual.values, color="#1f77b4", lw=1.8, label="Actual")
    
    ax.fill_between(
        forecast.index, lower.values, upper.values, color="red", alpha=0.08, linewidth=0
    )
    ax.plot(forecast.index, forecast.values, color="red", lw=2.0, label="Forecast")

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("EIA Net Generation — ETS forecast Jan–Aug 2025")
    ax.legend(loc="best")

    if not history.empty:
        ax.annotate(
            "History (2024)",
            xy=(history.index[-1], history.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#555555",
        )
    if not actual.empty:
        ax.annotate(
            "Actual (Jan–Aug 2025)",
            xy=(actual.index[-1], actual.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#1f77b4",
        )
    ax.annotate(
        "Forecast",
        xy=(forecast.index[-1], forecast.values[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        va="center",
        ha="left",
        color="red",
    )

    fig.tight_layout()
    save_plot(fig, config.ets_plot, dpi=300)
    plt.close(fig)
    print(f" ETS plot saved -> {config.ets_plot}")


def plot_generation_comparison(series: pd.Series, config: Config) -> None:
    """Plot comparison between ETS and SARIMAX."""
    ets_model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=config.season,
    ).fit(optimized=True)
    
    sarimax_model = sm.tsa.statespace.SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, config.season),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    history_end = config.history_end
    forecast_index = pd.date_range(
        history_end + pd.offsets.MonthBegin(1), periods=config.horizon, freq="MS"
    )
    ets_forecast = ets_model.forecast(config.horizon)
    sarimax_forecast = sarimax_model.forecast(config.horizon)

    actual = series.loc[forecast_index[0] : forecast_index[-1]]
    history = series.loc[:history_end]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, label="History", color="#888888", lw=1.5)
    
    if not actual.empty:
        ax.plot(actual.index, actual.values, label="Actual", color="#444444", lw=1.8)
    
    ax.plot(
        forecast_index,
        ets_forecast.values,
        label="ETS forecast",
        color="#d62728",
        lw=2.0,
    )
    ax.plot(
        forecast_index,
        sarimax_forecast.values,
        label="SARIMAX forecast",
        color="#1f77b4",
        lw=2.0,
    )

    ax.set_title("ETS vs SARIMAX — last fold comparison")
    ax.set_xlabel("")
    ax.grid(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_plot(fig, config.comparison_plot, dpi=300)
    plt.close(fig)
    print(f" ETS vs SARIMAX plot saved -> {config.comparison_plot}")


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass for this template
    config = parse_config(config_dict, script_dir)
    
    # Load series using consolidated loader
    series = load_series(config)
    print(f"Loaded {len(series)} data points")
    
    # Rolling origin evaluation
    _, last_true, last_pred = rolling_origin_ets(series, config)
    
    # Plot ETS forecast
    if last_pred is not None:
        plot_ets_forecast(series, config, last_pred)
    
    # Plot comparison
    plot_generation_comparison(series, config)
    
    print("\n Exponential smoothing analysis complete")


if __name__ == "__main__":
    main()
