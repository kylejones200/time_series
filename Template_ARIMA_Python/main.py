#!/usr/bin/env python3
"""Classical ARIMA baselines aligned with the 2025-11-08 article assets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import signalplot
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
    max_lag: int
    output_dir: Path
    uni_multi_plot: Path
    baseline_plot: Path
    ensemble_plot: Path
    streaming_plot: Path


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
        max_lag=int(cfg["evaluation"]["max_lag"]),
        output_dir=output_dir,
        uni_multi_plot=output_dir / cfg["output"]["uni_multi_plot"],
        baseline_plot=output_dir / cfg["output"]["baseline_plot"],
        ensemble_plot=output_dir / cfg["output"]["ensemble_plot"],
        streaming_plot=output_dir / cfg["output"]["streaming_plot"],
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


def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    month = df.index.month.values
    df["sin12"] = np.sin(2 * np.pi * month / 12.0)
    df["cos12"] = np.cos(2 * np.pi * month / 12.0)
    month_dummies = pd.get_dummies(month, prefix="month")
    df = df.join(month_dummies)
    return df


def rolling_origin_uni_vs_multi(
    series: pd.Series, config: Config
) -> Tuple[float, float, pd.Series, pd.Series, pd.Series]:
    idx = np.arange(len(series))
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    uni_maes = []
    mul_maes = []
    last_true = last_uni = last_mul = None

    for train_idx, _ in splitter.split(idx):
        end = train_idx[-1]
        train = series.iloc[: end + 1]
        future = series.iloc[end + 1 : end + 1 + config.horizon]
        if future.empty:
            continue

        uni_model = sm.tsa.statespace.SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, config.season),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        uni_forecast = uni_model.forecast(len(future))
        uni_maes.append(mean_absolute_error(future.values, uni_forecast.values))

        X_train = make_calendar_features(train.index)
        X_future = make_calendar_features(future.index)
        multi_model = sm.tsa.statespace.SARIMAX(
            train,
            exog=X_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, config.season),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        mul_forecast = multi_model.forecast(len(future), exog=X_future)
        mul_maes.append(mean_absolute_error(future.values, mul_forecast.values))

        last_true = future
        last_uni = uni_forecast
        last_mul = mul_forecast

    print(f"SARIMAX univariate MAE: {np.mean(uni_maes):.3f}")
    print(f"SARIMAX with exogenous features MAE: {np.mean(mul_maes):.3f}")
    return np.mean(uni_maes), np.mean(mul_maes), last_true, last_uni, last_mul


def plot_uni_vs_multi(
    series: pd.Series,
    config: Config,
    future: pd.Series,
    uni: pd.Series,
    multi: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(series.index, series.values, label="History", alpha=0.6)
    ax.plot(uni.index, uni.values, label="Univariate SARIMAX last fold")
    ax.plot(multi.index, multi.values, label="SARIMAX + calendar last fold")
    ax.legend(frameon=False)
    ax.set_title("Univariate vs exogenous SARIMAX — last fold")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(config.uni_multi_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Univariate vs exogenous plot saved -> {config.uni_multi_plot}")


def rolling_origin_linear_baseline(
    series: pd.Series, config: Config
) -> Tuple[float, pd.Series]:
    idx = np.arange(len(series))
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    maes = []
    last_pred = None

    def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
        df = pd.DataFrame(index=index)
        df["sin12"] = np.sin(2 * np.pi * df.index.month / 12.0)
        df["cos12"] = np.cos(2 * np.pi * df.index.month / 12.0)
        for k in range(1, config.season + 1):
            df[f"m{k}"] = (df.index.month == k).astype(int)
        return df.values

    X_full = calendar_features(series.index)
    for train_idx, _ in splitter.split(idx):
        end = train_idx[-1]
        y_train = series.iloc[: end + 1]
        y_future = series.iloc[end + 1 : end + 1 + config.horizon]
        if y_future.empty:
            continue
        model = LinearRegression()
        model.fit(X_full[: end + 1], y_train.values)
        preds = model.predict(X_full[end + 1 : end + 1 + config.horizon])
        maes.append(mean_absolute_error(y_future.values, preds))
        last_pred = pd.Series(preds, index=y_future.index)

    print(f"Linear calendar baseline MAE: {np.mean(maes):.3f}")
    return np.mean(maes), last_pred


def plot_baseline(series: pd.Series, config: Config, predictions: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(series.index, series.values, label="History", alpha=0.6)
    if predictions is not None:
        ax.plot(
            predictions.index, predictions.values, label="Linear baseline last fold"
        )
    ax.legend(frameon=False)
    ax.set_title("Linear calendar baseline — last fold")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(config.baseline_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Baseline plot saved -> {config.baseline_plot}")


def rolling_origin_ensemble(
    series: pd.Series, config: Config
) -> Tuple[float, float, float, pd.Series, dict]:
    idx = np.arange(len(series))
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    ets_maes, sar_maes, ens_maes = [], [], []
    last_components = {}
    last_true = None

    for train_idx, _ in splitter.split(idx):
        end = train_idx[-1]
        train = series.iloc[: end + 1]
        future = series.iloc[end + 1 : end + 1 + config.horizon]
        if future.empty:
            continue

        ets = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=config.season
        ).fit(optimized=True)
        ets_fore = ets.forecast(len(future))
        sar = sm.tsa.statespace.SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, config.season),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        sar_fore = sar.forecast(len(future))
        ensemble_fore = (ets_fore.values + sar_fore.values) / 2.0

        ets_maes.append(mean_absolute_error(future.values, ets_fore.values))
        sar_maes.append(mean_absolute_error(future.values, sar_fore.values))
        ens_maes.append(mean_absolute_error(future.values, ensemble_fore))
        last_true = future
        last_components = {
            "ETS": ets_fore,
            "SARIMAX": sar_fore,
            "Ensemble": pd.Series(ensemble_fore, index=future.index),
        }

    print(f"ETS MAE: {np.mean(ets_maes):.3f}")
    print(f"SARIMAX MAE: {np.mean(sar_maes):.3f}")
    print(f"Ensemble MAE: {np.mean(ens_maes):.3f}")
    return (
        np.mean(ets_maes),
        np.mean(sar_maes),
        np.mean(ens_maes),
        last_true,
        last_components,
    )


def plot_ensemble(
    series: pd.Series, config: Config, future: pd.Series, components: dict
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(series.index, series.values, label="History", alpha=0.6)
    for name, preds in components.items():
        ax.plot(preds.index, preds.values, label=f"{name} last fold")
    ax.legend(frameon=False)
    ax.set_title("ETS vs SARIMAX vs ensemble — last fold")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(config.ensemble_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Ensemble plot saved -> {config.ensemble_plot}")


def online_sarimax_forecast(
    series: pd.Series, config: Config
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    history_window = deque(maxlen=60)
    times, y_true, y_hat = [], [], []

    for timestamp, value in series.items():
        if len(history_window) >= 24:
            history_series = pd.Series(list(history_window))
            model = sm.tsa.statespace.SARIMAX(
                history_series,
                order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            pred = float(model.forecast(steps=1).iloc[0])
            times.append(timestamp)
            y_hat.append(pred)
            y_true.append(value)
        history_window.append(value)

    return (
        pd.Series(y_true, index=pd.to_datetime(times)),
        pd.Series(y_hat, index=pd.to_datetime(times)),
        series,
    )


def plot_streaming(
    series: pd.Series, config: Config, truth: pd.Series, preds: pd.Series
) -> None:
    history_start = pd.Timestamp("2024-01-01")
    history_end = pd.Timestamp("2024-12-01")
    history = series.loc[history_start:history_end]
    actual = series.loc[preds.index.min() : preds.index.max()]

    residuals = truth.align(preds, join="inner")
    residuals_series = residuals[0] - residuals[1]
    sigma = float(residuals_series.std(ddof=1)) if len(residuals_series) else 0.0
    upper = preds + 1.96 * sigma
    lower = preds - 1.96 * sigma

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, color="#888888", lw=1.5)
    ax.axvline(pd.Timestamp("2025-01-01"), color="#666666", linestyle="--", lw=1)
    if not actual.empty:
        ax.plot(actual.index, actual.values, color="#444444", lw=1.8)
    ax.fill_between(
        preds.index,
        lower.values,
        upper.values,
        color="#000000",
        alpha=0.06,
        linewidth=0,
    )
    ax.plot(preds.index, preds.values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("Online SARIMAX one-step forecast Jan–Aug 2025")

    fig.tight_layout()
    fig.savefig(config.streaming_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Streaming plot saved -> {config.streaming_plot}")


def main() -> None:
    config = load_config()
    series = load_series(config)

    _, _, future, uni_pred, multi_pred = rolling_origin_uni_vs_multi(series, config)
    if future is not None and uni_pred is not None and multi_pred is not None:
        plot_uni_vs_multi(series, config, future, uni_pred, multi_pred)

    _, baseline_pred = rolling_origin_linear_baseline(series, config)
    if baseline_pred is not None:
        plot_baseline(series, config, baseline_pred)

    _, _, _, ensemble_truth, components = rolling_origin_ensemble(series, config)
    if components:
        plot_ensemble(series, config, ensemble_truth, components)

    truth_series, preds_series, _ = online_sarimax_forecast(series, config)
    plot_streaming(series, config, truth_series, preds_series)


if __name__ == "__main__":
    main()
