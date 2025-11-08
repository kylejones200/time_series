#!/usr/bin/env python3
"""StatsForecast AutoARIMA template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utilsforecast.losses import mae, mape, mse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_plot_style, apply_legend, save_plot


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    season_length: int
    prediction_length: int
    holdout_length: int
    output_dir: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg['data']['input_file']
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    model_cfg = cfg['model']
    return Config(
        data_path=data_path,
        date_col=cfg['data']['date_col'],
        value_col=cfg['data']['value_col'],
        freq=cfg['data'].get('freq', 'H'),
        season_length=model_cfg.get('season_length', 24),
        prediction_length=model_cfg['prediction_length'],
        holdout_length=model_cfg.get('holdout_length', model_cfg['prediction_length']),
        output_dir=output_dir,
    )


def load_series(config: Config) -> pd.DataFrame:
    df = pd.read_csv(config.data_path)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not found in CSV")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors='coerce')
    df = df.dropna(subset=[config.date_col, config.value_col])
    df = df.sort_values(config.date_col)

    df = df.reset_index(drop=True)
    df = df[[config.date_col, config.value_col]].rename(columns={
        config.date_col: 'ds',
        config.value_col: 'y'
    })
    return df


def prepare_data(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_resampled = df.set_index('ds').resample(config.freq)['y'].mean().reset_index()
    df_resampled['unique_id'] = 'series_1'

    if config.holdout_length >= len(df_resampled):
        raise ValueError("Holdout length must be smaller than series length")

    train = df_resampled.iloc[:-config.holdout_length].copy()
    test = df_resampled.iloc[-config.holdout_length:].copy()
    return train, test


def fit_and_forecast(train: pd.DataFrame, test: pd.DataFrame, config: Config) -> pd.DataFrame:
    models = [AutoARIMA(season_length=config.season_length)]
    sf = StatsForecast(models=models, freq=config.freq, n_jobs=-1)
    sf.fit(train)
    forecasts = sf.forecast(h=config.prediction_length, df=train)
    return forecasts


def compute_metrics(test: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    merged = test[['ds', 'y']].merge(forecast[['ds', 'AutoARIMA']], on='ds', how='left')
    actual = merged['y'].to_numpy()
    predicted = merged['AutoARIMA'].to_numpy()

    metrics = {
        'MAE': float(mae(actual, predicted)),
        'MSE': float(mse(actual, predicted)),
        'RMSE': float(np.sqrt(mse(actual, predicted))),
        'MAPE': float(mape(actual, predicted)),
    }
    return metrics, merged


def plot_forecast(full_df: pd.DataFrame, merged: pd.DataFrame, config: Config) -> None:
    fig, ax = setup_figure((12, 6), 150)
    apply_plot_style(ax, {'plotting': {
        'style': {'spines': {'top': False, 'right': False, 'bottom': True, 'left': True}, 'grid': False}
    }})

    ax.plot(full_df['ds'], full_df['y'], label='History', color='black')
    ax.plot(merged['ds'], merged['y'], label='Actual', color='green', linestyle='--')
    ax.plot(merged['ds'], merged['AutoARIMA'], label='AutoARIMA', color='tomato')

    ax.set_title('StatsForecast AutoARIMA')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, {'frameon': False, 'loc': 'best'})

    path = config.output_dir / 'statsforecast_forecast.png'
    save_plot(fig, path)
    plt.close(fig)
    print(f"✓ Forecast plot saved -> {path}")


def save_outputs(merged: pd.DataFrame, metrics: dict, config: Config) -> None:
    merged.rename(columns={'AutoARIMA': 'forecast'}, inplace=True)
    merged.to_csv(config.output_dir / 'statsforecast_forecast.csv', index=False)
    with open(config.output_dir / 'statsforecast_metrics.yaml', 'w') as f:
        yaml.safe_dump(metrics, f)
    print("✓ Forecast CSV and metrics saved")


def main():
    config = load_config()
    df = load_series(config)
    train, test = prepare_data(df, config)

    forecasts = fit_and_forecast(train, test, config)
    metrics, merged = compute_metrics(test, forecasts)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    plot_forecast(df, merged, config)
    save_outputs(merged, metrics, config)


if __name__ == "__main__":
    main()
