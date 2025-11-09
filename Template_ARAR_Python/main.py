#!/usr/bin/env python3
"""
ARAR: Autoregressive Autoregressive
Refactored to support config-driven workflows, evaluation, and comparison with ARIMA.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import apply_legend, save_plot, setup_figure
from utils.ts_utils import detect_frequency, load_ts_data, resample_ts, split_ts

warnings.filterwarnings("ignore")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Path | str = "config.yaml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_data_path(input_file: str) -> Path:
    path = Path(input_file)
    if path.is_absolute():
        return path
    candidate = repo_root() / "data" / path
    if candidate.exists():
        return candidate
    return path


def load_series(config: Dict) -> pd.Series:
    data_cfg = config["data"]
    series = load_ts_data(
        resolve_data_path(data_cfg["input_file"]),
        date_col=data_cfg.get("date_col", "date"),
        value_col=data_cfg.get("value_col", "value"),
    ).sort_index()

    resample_cfg = data_cfg.get("resample", {})
    if resample_cfg.get("enabled"):
        series = resample_ts(
            series,
            freq=resample_cfg.get("freq", "H"),
            method=resample_cfg.get("method", "mean"),
        )
    return series


def select_lags(series: pd.Series, model_cfg: Dict) -> List[int]:
    max_lag = int(model_cfg.get("max_lag", 20))
    lag_method = model_cfg.get("lag_method", "powers_of_2")
    acf_vals = acf(series, nlags=min(max_lag, len(series) - 1)) if len(series) > 1 else [1.0]

    if lag_method == "custom":
        lags = model_cfg.get("custom_lags", [1, 2, 4, 8])
    elif lag_method == "auto":
        threshold = model_cfg.get("acf_threshold", 0.1)
        lags = [i for i in range(1, min(max_lag + 1, len(series))) if abs(acf_vals[i]) > threshold]
    else:  # powers_of_2
        lags = []
        power = 0
        while True:
            lag = 2 ** power
            if lag > max_lag or lag >= len(series):
                break
            lags.append(lag)
            power += 1
        if 1 not in lags and len(series) > 1:
            lags = [1] + lags

    lags = sorted({lag for lag in lags if lag > 0})
    return lags or [1]


def difference_series(series: pd.Series) -> pd.Series:
    return series.diff().dropna()


def fit_arar(series: pd.Series, lags: List[int]) -> AutoReg:
    return AutoReg(series, lags=lags, old_names=False).fit()


def generate_forecast(
    model: AutoReg,
    transformed_series: pd.Series,
    model_cfg: Dict,
    base_level: float,
    horizon: int,
) -> pd.Series:
    preds = model.predict(start=len(transformed_series), end=len(transformed_series) + horizon - 1)
    forecast = pd.Series(preds)
    if model_cfg.get("differenced", True):
        forecast = forecast.cumsum() + base_level
    return forecast


def compute_metrics(actual: pd.Series, forecast: pd.Series) -> Dict[str, float]:
    mae = mean_absolute_error(actual, forecast)
    rmse = mean_squared_error(actual, forecast, squared=False)
    mape = mean_absolute_percentage_error(actual, forecast)
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def save_metrics(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(metrics, fh, sort_keys=True)


def plot_historical_and_diff(
    original: pd.Series,
    differenced: pd.Series,
    output_dir: Path,
    config: Dict,
) -> None:
    fig, axes = setup_figure(config, nrows=2, ncols=1)
    axes[0].plot(original.index, original.values, color=config["plotting"]["style"]["colors"]["primary"])
    axes[0].set_title("Original Series")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Value")

    axes[1].plot(differenced.index, differenced.values, color=config["plotting"]["style"]["colors"]["secondary"])
    axes[1].set_title("Differenced Series")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Difference")

    plt.tight_layout()
    if config["output"].get("save_plots", True):
        save_plot(fig, output_dir / config["output"]["plots"]["historical"], config)
    plt.close(fig)


def plot_forecast_vs_actual(
    train: pd.Series,
    actual: pd.Series,
    forecast: pd.Series,
    output_dir: Path,
    config: Dict,
) -> None:
    fig, ax = setup_figure(config)
    ax.plot(train.index, train.values, label="Historical", color=config["plotting"]["style"]["colors"]["primary"])
    ax.plot(actual.index, actual.values, label="Actual", color=config["plotting"]["style"]["colors"]["accent"])
    ax.plot(forecast.index, forecast.values, label="ARAR Forecast", color=config["plotting"]["style"]["colors"]["secondary"])

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("ARAR Forecast vs Actuals")
    apply_legend(ax, config)

    plt.tight_layout()
    if config["output"].get("save_plots", True):
        save_plot(fig, output_dir / config["output"]["plots"]["forecast"], config)
    plt.close(fig)


def plot_comparison(
    series: pd.Series,
    forecast_index: pd.Index,
    arar_forecast: pd.Series,
    metrics: Dict[str, Dict[str, float]],
    output_dir: Path,
    config: Dict,
    arima_forecast: pd.Series | None = None,
) -> None:
    fig, ax = setup_figure(config)
    ax.plot(series.index, series.values, label="Historical", color=config["plotting"]["style"]["colors"]["primary"])
    ax.plot(forecast_index, arar_forecast.values, label="ARAR Forecast", linestyle="--", color=config["plotting"]["style"]["colors"]["secondary"])

    if arima_forecast is not None:
        ax.plot(forecast_index, arima_forecast.values, label="ARIMA Forecast", linestyle=":", color=config["plotting"]["style"]["colors"]["accent"])

    title_parts = [f"ARAR MAPE: {metrics['arar']['mape']:.3f}"]
    if "arima" in metrics:
        title_parts.append(f"ARIMA MAPE: {metrics['arima']['mape']:.3f}")
    ax.set_title("Forecast Comparison\n" + " | ".join(title_parts))
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    apply_legend(ax, config)

    plt.tight_layout()
    if config["output"].get("save_plots", True):
        save_plot(fig, output_dir / config["output"]["plots"]["comparison"], config)
    plt.close(fig)


def main() -> None:
    config = load_config()
    series = load_series(config)

    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})
    horizon = int(eval_cfg.get("horizon", model_cfg.get("forecast_horizon", 12)))

    if horizon < 1:
        raise ValueError("Evaluation horizon must be at least 1.")
    if horizon >= len(series):
        raise ValueError("Evaluation horizon must be smaller than the length of the series.")

    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    if train.empty or test.empty:
        train, test = split_ts(series, train_size=model_cfg.get("train_size", 0.8))
        horizon = len(test)

    transformed_train = difference_series(train) if model_cfg.get("differenced", True) else train
    lags = select_lags(transformed_train, model_cfg)
    arar_model = fit_arar(transformed_train, lags)

    forecast_values = generate_forecast(
        arar_model,
        transformed_train,
        model_cfg,
        base_level=float(train.iloc[-1]),
        horizon=horizon,
    )

    forecast_index = test.index if len(test) == horizon else pd.date_range(
        start=train.index[-1], periods=horizon + 1, freq=detect_frequency(series)
    )[1:]
    arar_forecast = pd.Series(forecast_values.values, index=forecast_index)

    metrics: Dict[str, Dict[str, float]] = {"arar": compute_metrics(test.reindex(forecast_index), arar_forecast)}

    compare_cfg = eval_cfg.get("compare_arima", {})
    arima_forecast = None
    if compare_cfg.get("enabled", False):
        order = tuple(compare_cfg.get("order", (2, 1, 2)))
        arima_model = ARIMA(train, order=order).fit()
        arima_pred = arima_model.forecast(steps=horizon)
        arima_forecast = pd.Series(arima_pred.values, index=forecast_index)
        metrics["arima"] = compute_metrics(test.reindex(forecast_index), arima_forecast)

    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(metrics, output_dir)
    plot_historical_and_diff(train, transformed_train, output_dir, config)
    plot_forecast_vs_actual(train, test.reindex(forecast_index), arar_forecast, output_dir, config)
    plot_comparison(series, forecast_index, arar_forecast, metrics, output_dir, config, arima_forecast=arima_forecast)

    print("✓ ARAR forecasting complete")


if __name__ == "__main__":
    main()

