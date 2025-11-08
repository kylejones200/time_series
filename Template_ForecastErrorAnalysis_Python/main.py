#!/usr/bin/env python3
"""Forecast error diagnostics using seasonal ETS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_plot_style, apply_legend, save_plot


def simple_average_error(errors: pd.Series) -> float:
    return errors.mean()


def moving_average_error(errors: pd.Series, window: int) -> float:
    window = min(window, len(errors))
    return errors.iloc[-window:].mean()


def exponential_smoothing_error(errors: pd.Series, alpha: float = 0.2) -> pd.Series:
    smoothed = errors.ewm(alpha=alpha, adjust=False).mean()
    return smoothed


def sample_variance_error(errors: pd.Series, window: Optional[int] = None) -> float:
    subset = errors.iloc[-window:] if window is not None else errors
    return subset.var(ddof=1)


def mean_absolute_deviation(errors: pd.Series) -> float:
    return errors.abs().mean()


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    eps = 1e-8
    return (np.abs((y_true - y_pred) / (y_true + eps))).mean() * 100


def mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return ((y_true - y_pred) ** 2).mean()


def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


@dataclass
class Config:
    url: Optional[str]
    input_file: Optional[str]
    date_col: str
    value_col: str
    seasonal_periods: int
    moving_average_window: int
    exponential_alpha: float
    output_dir: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    data_cfg = cfg['data']
    url = data_cfg.get('url')
    input_file = data_cfg.get('input_file')

    if url is None and input_file is None:
        raise ValueError("Either 'url' or 'input_file' must be provided in config.")

    if input_file:
        input_path = repo_root / 'data' / input_file
        if not input_path.exists():
            raise FileNotFoundError(f"Local data file not found at {input_path}")

    return Config(
        url=url,
        input_file=input_file,
        date_col=data_cfg['date_col'],
        value_col=data_cfg['value_col'],
        seasonal_periods=cfg['model']['seasonal_periods'],
        moving_average_window=cfg['analysis']['moving_average_window'],
        exponential_alpha=cfg['analysis']['exponential_alpha'],
        output_dir=output_dir,
    )


def load_series(config: Config) -> pd.Series:
    if config.url:
        df = pd.read_csv(config.url)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        df = pd.read_csv(repo_root / 'data' / config.input_file)

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors='coerce')
    df = df.dropna(subset=[config.date_col, config.value_col])
    df = df.sort_values(config.date_col)
    df = df.set_index(config.date_col)

    series = pd.to_numeric(df[config.value_col], errors='coerce').dropna()
    return series


def fit_ets(series: pd.Series, seasonal_periods: int) -> pd.Series:
    model = ExponentialSmoothing(
        series,
        seasonal='add',
        trend='add',
        seasonal_periods=seasonal_periods,
        initialization_method="estimated"
    ).fit()
    return model.fittedvalues


def compute_metrics_dict(series: pd.Series, fitted: pd.Series, config: Config,
                         smoothed_errors: pd.Series) -> dict:
    errors = series - fitted
    metrics = {
        'simple_average_error': float(simple_average_error(errors)),
        'moving_average_error': float(moving_average_error(errors, config.moving_average_window)),
        'exp_smoothed_last': float(smoothed_errors.iloc[-1]),
        'variance': float(sample_variance_error(errors)),
        'mad': float(mean_absolute_deviation(errors)),
        'mape': float(mean_absolute_percentage_error(series, fitted)),
        'mse': float(mean_squared_error(series, fitted)),
        'rmse': float(root_mean_squared_error(series, fitted)),
    }
    return metrics


def plot_results(series: pd.Series, fitted: pd.Series, errors: pd.Series,
                 smoothed_errors: pd.Series, config: Config) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    apply_plot_style(axes[0], {'plotting': {
        'style': {'spines': {'top': False, 'right': False, 'bottom': True, 'left': True}, 'grid': False}
    }})
    axes[0].plot(series.index, series.values, color='black', label='Actual')
    axes[0].plot(fitted.index, fitted.values, color='tomato', label='ETS Fitted')
    axes[0].set_title('Actual vs Fitted')
    apply_legend(axes[0], {'frameon': False, 'loc': 'best'})

    apply_plot_style(axes[1], {'plotting': {'style': {'spines': {'top': False, 'right': False, 'bottom': True, 'left': True}, 'grid': False}}})
    axes[1].plot(errors.index, errors.values, color='royalblue', label='Errors')
    axes[1].axhline(0, color='grey', linewidth=1, linestyle='--')
    axes[1].set_title('Forecast Errors')

    apply_plot_style(axes[2], {'plotting': {'style': {'spines': {'top': False, 'right': False, 'bottom': True, 'left': True}, 'grid': False}}})
    axes[2].plot(smoothed_errors.index, smoothed_errors.values, color='seagreen', label='Exp Smoothed Errors')
    axes[2].set_title('Exponentially Smoothed Errors')
    axes[2].set_xlabel('Date')

    for ax in axes:
        apply_legend(ax, {'frameon': False, 'loc': 'best'})

    plt.tight_layout()
    plot_path = config.output_dir / 'forecast_error_analysis.png'
    save_plot(fig, plot_path)
    plt.close(fig)
    print(f"✓ Plot saved -> {plot_path}")


def save_metrics(metrics: dict, config: Config) -> None:
    metrics_path = config.output_dir / 'forecast_error_metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.safe_dump({k: float(v) for k, v in metrics.items()}, f)
    print(f"✓ Metrics saved -> {metrics_path}")


def main():
    config = load_config()
    series = load_series(config)
    fitted = fit_ets(series, config.seasonal_periods)
    errors = series - fitted
    smoothed_errors = exponential_smoothing_error(errors, config.exponential_alpha)

    metrics = compute_metrics_dict(series, fitted, config, smoothed_errors)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    plot_results(series, fitted, errors, smoothed_errors, config)
    save_metrics(metrics, config)


if __name__ == "__main__":
    main()
