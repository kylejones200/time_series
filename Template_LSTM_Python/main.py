#!/usr/bin/env python3
"""LSTM evaluation aligned with the 2025-11-08 article assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    horizon: int
    n_splits: int
    input_chunk_length: int
    output_chunk_length: int
    epochs: int
    output_dir: Path
    output_plot: Path


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
        horizon=int(cfg["model"]["horizon"]),
        n_splits=int(cfg["model"]["n_splits"]),
        input_chunk_length=int(cfg["model"]["input_chunk_length"]),
        output_chunk_length=int(cfg["model"]["output_chunk_length"]),
        epochs=int(cfg["model"]["epochs"]),
        output_dir=output_dir,
        output_plot=output_dir / cfg["output"]["tufte_plot"],
    )


def load_series(config: Config) -> TimeSeries:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")

    df = pd.read_csv(config.data_path)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not present in CSV")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.value_col])
    df = df.sort_values(config.date_col)
    series = pd.Series(
        pd.to_numeric(df[config.value_col], errors="coerce").dropna().values,
        index=pd.DatetimeIndex(df[config.date_col]),
    )
    series = series.asfreq(config.freq).astype(float)
    return TimeSeries.from_series(series)


def build_model(config: Config) -> RNNModel:
    return RNNModel(
        model="LSTM",
        input_chunk_length=config.input_chunk_length,
        output_chunk_length=config.output_chunk_length,
        training_length=max(config.input_chunk_length, 24),
        n_rnn_layers=2,
        hidden_dim=64,
        n_epochs=config.epochs,
        random_state=42,
        pl_trainer_kwargs={
            "enable_progress_bar": False,
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
        },
    )


def rolling_origin_lstm(series: TimeSeries, config: Config) -> Tuple[float, TimeSeries, TimeSeries]:
    values = series.to_series()
    idx = np.arange(len(values))
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    maes = []
    last_true = None
    last_pred = None

    for train_idx, _ in splitter.split(idx):
        end = train_idx[-1]
        train_ts = series.drop_after(series.time_index[end])
        future = series.split_after(series.time_index[end])[1]
        horizon_ts = future.drop_after(future.time_index[min(config.horizon - 1, len(future) - 1)])
        if len(horizon_ts) == 0:
            continue

        scaler = Scaler()
        train_scaled = scaler.fit_transform(train_ts)
        model = build_model(config)
        model.fit(train_scaled)
        forecast_scaled = model.predict(len(horizon_ts))
        forecast = scaler.inverse_transform(forecast_scaled)
        mae = mean_absolute_error(horizon_ts.values().ravel(), forecast.values().ravel())
        maes.append(mae)
        last_true = horizon_ts
        last_pred = forecast

    mean_mae = float(np.mean(maes)) if maes else float("nan")
    print(f"LSTM rolling-origin MAE: {mean_mae:.3f}")
    return mean_mae, last_true, last_pred


def plot_tufte(series: TimeSeries, config: Config, forecast: TimeSeries) -> None:
    s = series.to_series()
    start_2024 = pd.Period("2024-01", freq="M").start_time + pd.offsets.MonthBegin(0)
    end_2024 = pd.Period("2024-12", freq="M").start_time + pd.offsets.MonthBegin(0)
    jan_2025 = pd.Period("2025-01", freq="M").start_time + pd.offsets.MonthBegin(0)
    aug_2025 = pd.Period("2025-08", freq="M").start_time + pd.offsets.MonthBegin(0)

    history = s.loc[start_2024:end_2024]
    actual = s.loc[jan_2025:aug_2025]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, color="#888888", lw=1.5)
    ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
    if not actual.empty:
        ax.plot(actual.index, actual.values, color="#444444", lw=1.8)

    plt_forecast = forecast
    ax.fill_between(
        plt_forecast.time_index,
        plt_forecast.values().ravel() - 1.96 * np.std(plt_forecast.values().ravel()),
        plt_forecast.values().ravel() + 1.96 * np.std(plt_forecast.values().ravel()),
        color="#000000",
        alpha=0.06,
        linewidth=0,
    )
    ax.plot(plt_forecast.time_index, plt_forecast.values().ravel(), color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("EIA Net Generation — LSTM forecast Jan–Aug 2025")

    if not history.empty:
        ax.annotate(
            "History (2024)",
            xy=(history.index[-1], history.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#666666",
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
            color="#444444",
        )
    ax.annotate(
        "LSTM forecast",
        xy=(plt_forecast.time_index[-1], plt_forecast.values().ravel()[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        va="center",
        ha="left",
        color="#000000",
    )

    fig.tight_layout()
    fig.savefig(config.output_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ LSTM plot saved -> {config.output_plot}")


def main() -> None:
    config = load_config()
    series = load_series(config)
    mean_mae, _, forecast = rolling_origin_lstm(series, config)
    if forecast is not None:
        plot_tufte(series, config, forecast)


if __name__ == "__main__":
    main()

