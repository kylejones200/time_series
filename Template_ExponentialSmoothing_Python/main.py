#!/usr/bin/env python3
"""Exponential smoothing evaluations aligned with the 2025-11-08 article assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import signalplot
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm


@dataclass
class Config:
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


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg["data"]["input_file"]
    output_dir = Path(__file__).parent / cfg["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    experiment = cfg["experiment"]

    return Config(
        data_path=data_path,
        date_col=cfg["data"]["date_col"],
        value_col=cfg["data"]["value_col"],
        freq=cfg["data"].get("freq", "MS"),
        history_end=pd.Timestamp(experiment["history_end"]),
        forecast_start=pd.Timestamp(experiment["forecast_start"]),
        forecast_end=pd.Timestamp(experiment["forecast_end"]),
        horizon=int(cfg["evaluation"]["horizon"]),
        n_splits=int(cfg["evaluation"]["n_splits"]),
        season=int(cfg["evaluation"]["season"]),
        output_dir=output_dir,
        ets_plot=output_dir / cfg["output"]["ets_plot"],
        comparison_plot=output_dir / cfg["output"]["comparison_plot"],
    )


def load_series(config: Config) -> pd.Series:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")

    df = pd.read_csv(config.data_path)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not present in CSV")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.value_col])
    df = df.sort_values(config.date_col).set_index(config.date_col)
    series = pd.to_numeric(df[config.value_col], errors="coerce").dropna()
    return series.asfreq(config.freq).astype(float)


def rolling_origin_ets(
    series: pd.Series, config: Config
) -> Tuple[float, pd.Series, pd.Series]:
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


def plot_ets_tufte(series: pd.Series, config: Config, last_forecast: pd.Series) -> None:
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
    ax.plot(history.index, history.values, color="#555555", lw=1.5)
    ax.axvline(config.forecast_start, color="#777777", linestyle="--", lw=1)
    if not actual.empty:
        ax.plot(actual.index, actual.values, color="#1f77b4", lw=1.8)
    ax.fill_between(
        forecast.index, lower.values, upper.values, color="red", alpha=0.08, linewidth=0
    )
    ax.plot(forecast.index, forecast.values, color="red", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

# Apply SignalPlot's clean defaults
signalplot.apply()

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("EIA Net Generation — ETS forecast Jan–Aug 2025")

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
    fig.savefig(config.ets_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ ETS plot saved -> {config.ets_plot}")


def plot_generation_comparison(series: pd.Series, config: Config) -> None:
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
    fig.savefig(config.comparison_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ ETS vs SARIMAX plot saved -> {config.comparison_plot}")


def main() -> None:
    config = load_config()
    series = load_series(config)

    _, last_true, last_pred = rolling_origin_ets(series, config)
    if last_pred is not None:
        plot_ets_tufte(series, config, last_pred)
    plot_generation_comparison(series, config)


if __name__ == "__main__":
    main()
