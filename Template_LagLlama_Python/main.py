#!/usr/bin/env python3
"""Lag-Llama forecasting template.

Requires the lag-llama package (install via
`pip install git+https://github.com/time-series-foundation-models/lag-llama`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from lag_llama.gluon.estimator import LagLlamaEstimator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_plot_style, apply_legend, save_plot


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    context_length: int
    prediction_length: int
    rope_scaling: bool
    ckpt_path: Path
    output_dir: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg['data']['input_file']
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    ckpt = Path(cfg['model']['checkpoint'])
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Lag-Llama checkpoint not found at {ckpt}. "
            "Download from https://huggingface.co/time-series-foundation-models/Lag-Llama"
        )

    return Config(
        data_path=data_path,
        date_col=cfg['data']['date_col'],
        value_col=cfg['data']['value_col'],
        freq=cfg['data']['freq'],
        context_length=cfg['model']['context_length'],
        prediction_length=cfg['model']['prediction_length'],
        rope_scaling=cfg['model'].get('rope_scaling', False),
        ckpt_path=ckpt,
        output_dir=output_dir,
    )


def load_series(config: Config) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")

    df = pd.read_csv(config.data_path)
    df[config.date_col] = pd.to_datetime(df[config.date_col], errors='coerce')
    df = df.dropna(subset=[config.date_col, config.value_col])
    df = df.sort_values(config.date_col)
    df = df.rename(columns={config.date_col: 'timestamp', config.value_col: 'value'})
    df = df.set_index('timestamp').asfreq(config.freq).reset_index()
    df['item_id'] = 'series_1'
    return df[['item_id', 'timestamp', 'value']]


def build_dataset(df: pd.DataFrame) -> PandasDataset:
    return PandasDataset({
        'target': df.pivot(index='timestamp', columns='item_id', values='value')
    })


def load_estimator(config: Config) -> LagLlamaEstimator:
    ckpt = torch.load(config.ckpt_path, map_location='cpu', weights_only=False)
    kwargs = ckpt["hyper_parameters"]["model_kwargs"]

    rope_args = None
    if config.rope_scaling:
        rope_args = {
            'type': 'linear',
            'factor': max(1.0, (config.context_length + config.prediction_length) / kwargs['context_length'])
        }

    estimator = LagLlamaEstimator(
        ckpt_path=str(config.ckpt_path),
        prediction_length=config.prediction_length,
        context_length=config.context_length,
        input_size=kwargs['input_size'],
        n_layer=kwargs['n_layer'],
        n_embd_per_head=kwargs['n_embd_per_head'],
        n_head=kwargs['n_head'],
        scaling=kwargs['scaling'],
        time_feat=kwargs['time_feat'],
        rope_scaling=rope_args,
    )
    return estimator


def forecast(estimator: LagLlamaEstimator, dataset: PandasDataset, config: Config) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    predictor = estimator.create_predictor(batch_size=1)

    train_data, test_template = split(dataset, offset=-config.prediction_length)
    test_instances = test_template.generate_instances(prediction_length=config.prediction_length, windows=1)

    forecasts = list(predictor.predict(test_instances.input))
    forecast = forecasts[0].samples.mean(axis=0)

    label = list(test_instances.label)[0][0].values
    index = list(test_instances.label)[0][0].index
    return label, forecast, index


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}


def plot_forecast(full_df: pd.DataFrame, forecast_index: pd.DatetimeIndex,
                  actual: np.ndarray, predicted: np.ndarray,
                  metrics: dict, config: Config) -> None:
    fig, ax = setup_figure((12, 5), 150)
    apply_plot_style(ax, {'plotting': {
        'style': {'spines': {'top': False, 'right': False, 'bottom': True, 'left': True}, 'grid': False}
    }})

    series = full_df.set_index('timestamp')['value']
    ax.plot(series.index, series.values, color='black', label='History')
    ax.plot(forecast_index, actual, color='green', linestyle='--', label='Actual')
    ax.plot(forecast_index, predicted, color='tomato', label='Lag-Llama forecast')

    metrics_text = '\n'.join(f"{k}: {v:.3f}" for k, v in metrics.items())
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    apply_legend(ax, {'frameon': False, 'loc': 'best'})
    ax.set_title('Lag-Llama Forecast')
    ax.set_xlabel('Date')

    path = config.output_dir / 'lag_llama_forecast.png'
    save_plot(fig, path)
    plt.close(fig)
    print(f"✓ Plot saved -> {path}")


def save_outputs(forecast_index: pd.DatetimeIndex, actual: np.ndarray,
                 predicted: np.ndarray, metrics: dict, config: Config) -> None:
    df = pd.DataFrame({'timestamp': forecast_index, 'actual': actual, 'forecast': predicted})
    csv_path = config.output_dir / 'lag_llama_forecast.csv'
    df.to_csv(csv_path, index=False)

    metrics_path = config.output_dir / 'lag_llama_metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.safe_dump(metrics, f)

    print(f"✓ Forecast CSV -> {csv_path}")
    print(f"✓ Metrics YAML -> {metrics_path}")


def main():
    config = load_config()
    df = load_series(config)
    dataset = build_dataset(df)
    estimator = load_estimator(config)

    actual, forecast, index = forecast(estimator, dataset, config)
    metrics = compute_metrics(actual, forecast)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    plot_forecast(df, index, actual, forecast, metrics, config)
    save_outputs(index, actual, forecast, metrics, config)


if __name__ == "__main__":
    main()
