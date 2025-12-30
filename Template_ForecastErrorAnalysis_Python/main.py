#!/usr/bin/env python3
"""Seasonal naive error diagnostics aligned with the 2025-11-08 article assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import signalplot
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Apply SignalPlot's clean defaults
signalplot.apply()


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    horizon: int
    n_splits: int
    season: int
    output_dir: Path
    error_plot: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg["data"]["input_file"]
    output_dir = Path(__file__).parent / cfg["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    return Config(
        data_path=data_path,
        date_col=cfg["data"]["date_col"],
        value_col=cfg["data"]["value_col"],
        freq=cfg["data"].get("freq", "MS"),
        horizon=int(cfg["evaluation"]["horizon"]),
        n_splits=int(cfg["evaluation"]["n_splits"]),
        season=int(cfg["evaluation"]["season"]),
        output_dir=output_dir,
        error_plot=output_dir / cfg["output"]["error_plot"],
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


def mase_denominator(train: pd.Series, season: int) -> float:
    diffs = np.abs(train.values[season:] - train.values[:-season])
    if len(diffs) == 0 or np.allclose(diffs, 0.0):
        return 1.0
    return float(np.mean(diffs))


def seasonal_naive_forecast(train: pd.Series, horizon: int, season: int) -> np.ndarray:
    forecast = []
    values = train.values
    for i in range(horizon):
        src_idx = len(values) - season + i
        if src_idx >= 0:
            forecast.append(values[src_idx])
        else:
            forecast.append(values[-1])
    return np.asarray(forecast, dtype=float)


def rolling_origin_metrics(
    series: pd.Series, config: Config
) -> Tuple[List[dict], pd.Series, pd.Series]:
    idx = np.arange(len(series))
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    metrics: List[dict] = []
    last_truth = None
    last_forecast = None

    for train_idx, _ in splitter.split(idx):
        end = train_idx[-1]
        train = series.iloc[: end + 1]
        future = series.iloc[end + 1 : end + 1 + config.horizon]
        if future.empty:
            continue

        forecast = seasonal_naive_forecast(train, len(future), config.season)

        mae = mean_absolute_error(future.values, forecast)
        y = future.values.astype(float)
        denom = np.where(y == 0, np.finfo(float).eps, y)
        mape = np.mean(np.abs((y - forecast) / denom)) * 100.0
        smape = (
            np.mean(
                2
                * np.abs(forecast - y)
                / (np.abs(y) + np.abs(forecast) + np.finfo(float).eps)
            )
            * 100.0
        )
        mase = np.mean(np.abs(y - forecast)) / mase_denominator(train, config.season)

        metrics.append({"MAE": mae, "MAPE": mape, "SMAPE": smape, "MASE": mase})
        last_truth = future
        last_forecast = pd.Series(forecast, index=future.index)

    return metrics, last_truth, last_forecast


def plot_last_fold(
    series: pd.Series, truth: pd.Series, forecast: pd.Series, config: Config
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(series.index, series.values, label="History", alpha=0.6)
    if truth is not None and forecast is not None:
        ax.plot(truth.index, truth.values, label="Actual", color="#444444")
        ax.plot(
            forecast.index,
            forecast.values,
            label="Seasonal naive forecast",
            color="#d62728",
        )
    ax.legend(frameon=False)
    ax.set_title("Seasonal naive — last fold errors")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(config.error_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Error plot saved -> {config.error_plot}")


def main() -> None:
    config = load_config()
    series = load_series(config)

    metrics, truth, forecast = rolling_origin_metrics(series, config)
    mean_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
    print("Seasonal naive error summary:")
    for key, value in mean_metrics.items():
        print(f"{key}: {value:.3f}")

    plot_last_fold(series, truth, forecast, config)


if __name__ == "__main__":
    main()
