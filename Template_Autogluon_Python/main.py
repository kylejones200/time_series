#!/usr/bin/env python3
"""
AutoGluon for Time Series Forecasting

- Loads a CSV (single or multi-item) and converts to TimeSeriesDataFrame
- Fits AutoGluon's TimeSeriesPredictor
- Evaluates on a hold-out window and saves forecasts/metrics/plots
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj


plotting_utils = repo_import("utils.plotting_utils")
setup_figure = plotting_utils.setup_figure
apply_plot_style = plotting_utils.apply_plot_style
apply_legend = plotting_utils.apply_legend
save_plot = plotting_utils.save_plot


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    item_id_col: Optional[str]
    default_item_id: str
    frequency: Optional[str]
    prediction_length: int
    holdout_length: int
    model_path: Path
    eval_metric: str
    presets: Optional[str]
    hyperparameters: dict
    num_val_windows: int
    save_leaderboard: bool
    output_dir: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg['data']['input_file']
    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {data_path}")

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    return Config(
        data_path=data_path,
        date_col=cfg['data']['date_col'],
        value_col=cfg['data']['value_col'],
        item_id_col=cfg['data'].get('item_id_col'),
        default_item_id=cfg['data'].get('default_item_id', 'series_1'),
        frequency=cfg['data'].get('frequency'),
        prediction_length=cfg['model']['prediction_length'],
        holdout_length=cfg['model'].get('holdout_length', cfg['model']['prediction_length']),
        model_path=Path(__file__).parent / cfg['model'].get('model_path', 'autogluon_model'),
        eval_metric=cfg['model'].get('eval_metric', 'MAPE'),
        presets=cfg['model'].get('presets'),
        hyperparameters=cfg['model'].get('hyperparameters', {}),
        num_val_windows=cfg['model'].get('num_val_windows', 1),
        save_leaderboard=cfg['model'].get('save_leaderboard', True),
        output_dir=output_dir,
    )


def load_timeseries_dataframe(config: Config) -> TimeSeriesDataFrame:
    df = pd.read_csv(config.data_path)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified date/value columns not found in CSV")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors='coerce')
    df = df.dropna(subset=[config.date_col, config.value_col])

    if config.item_id_col and config.item_id_col in df.columns:
        id_col = config.item_id_col
    else:
        id_col = 'item_id'
        df[id_col] = config.default_item_id

    df = df[[id_col, config.date_col, config.value_col]].rename(columns={
        id_col: 'item_id',
        config.date_col: 'timestamp',
        config.value_col: 'target'
    })

    ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='item_id', timestamp_column='timestamp')
    if config.frequency:
        ts_df = ts_df.to_regular(freq=config.frequency)
    return ts_df


def split_train_test(ts_df: TimeSeriesDataFrame, holdout_length: int) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
    if holdout_length >= ts_df.num_timesteps_per_item().min():
        raise ValueError("Holdout length must be smaller than the shortest series length")
    train = ts_df.slice_by_timestep(None, -holdout_length)
    test = ts_df.slice_by_timestep(-holdout_length, None)
    return train, test


def train_predictor(train_ts: TimeSeriesDataFrame, config: Config) -> TimeSeriesPredictor:
    predictor = TimeSeriesPredictor(
        target='target',
        prediction_length=config.prediction_length,
        eval_metric=config.eval_metric,
        path=str(config.model_path),
    )
    fit_kwargs = {
        'train_data': train_ts,
        'hyperparameters': config.hyperparameters,
        'num_val_windows': config.num_val_windows,
    }
    if config.presets:
        fit_kwargs['presets'] = config.presets
    predictor.fit(**fit_kwargs)
    return predictor


def evaluate_and_save(predictor: TimeSeriesPredictor, train_ts: TimeSeriesDataFrame,
                      test_ts: TimeSeriesDataFrame, config: Config) -> None:
    forecasts = predictor.predict(test_ts)
    metrics = predictor.evaluate(test_ts)

    if config.save_leaderboard:
        leaderboard = predictor.leaderboard(train_ts, silent=True)
        leaderboard.to_csv(config.output_dir / 'autogluon_leaderboard.csv', index=False)

    # Save metrics
    metrics_path = config.output_dir / 'autogluon_metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.safe_dump({k: float(v) for k, v in metrics.items()}, f)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save forecast CSV
    forecast_df = forecasts.reset_index()
    forecast_df.to_csv(config.output_dir / 'autogluon_forecast.csv', index=False)

    plot_forecast(train_ts, test_ts, forecasts, config)


def plot_forecast(train_ts: TimeSeriesDataFrame, test_ts: TimeSeriesDataFrame,
                  forecasts: TimeSeriesDataFrame, config: Config) -> None:
    item_id = forecasts.index.levels[0][0]
    history = train_ts.loc[item_id]
    truth = test_ts.loc[item_id]
    pred = forecasts.loc[item_id]['mean']

    fig, ax = setup_figure((12, 6), 150)
    apply_plot_style(ax, {'plotting': {
        'style': {'spines': {'top': False, 'right': False, 'bottom': True, 'left': True}, 'grid': False}
    }})

    ax.plot(history.index, history.values, label='History', color='black')
    ax.plot(truth.index, truth.values, label='Actual', color='green', linestyle='--')
    ax.plot(truth.index, pred.values, label='Forecast', color='tomato')

    ax.set_title(f'AutoGluon Forecast (item={item_id})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, {'frameon': False, 'loc': 'best'})

    plot_path = config.output_dir / 'autogluon_forecast.png'
    save_plot(fig, plot_path)
    plt.close(fig)
    print(f"✓ Forecast plot saved -> {plot_path}")


def main():
    config = load_config()
    ts_df = load_timeseries_dataframe(config)
    train_ts, test_ts = split_train_test(ts_df, config.holdout_length)

    predictor = train_predictor(train_ts, config)
    evaluate_and_save(predictor, train_ts, test_ts, config)


if __name__ == "__main__":
    main()

