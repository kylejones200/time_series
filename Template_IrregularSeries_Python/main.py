#!/usr/bin/env python3
"""Irregular time series resampling and interpolation demos."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import yaml


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
save_plot = plotting_utils.save_plot


@dataclass
class Config:
    resample_rule: str
    start: str
    freq: str
    n_points: int
    gap_prob: float
    gp_length_scale: float
    output_dir: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    sim_cfg = cfg['simulation']
    gp_cfg = cfg['gaussian_process']
    return Config(
        resample_rule=cfg['resample']['rule'],
        start=sim_cfg['start'],
        freq=sim_cfg['freq'],
        n_points=sim_cfg['n_points'],
        gap_prob=sim_cfg['gap_probability'],
        gp_length_scale=gp_cfg['length_scale'],
        output_dir=output_dir,
    )


def simulate_irregular_series(config: Config) -> pd.Series:
    rng = np.random.default_rng(42)
    full_index = pd.date_range(config.start, periods=config.n_points, freq=config.freq)

    mask = rng.random(config.n_points) > config.gap_prob
    observed_index = full_index[mask]
    values = np.cumsum(rng.normal(loc=0.0, scale=1.0, size=mask.sum())) + 10

    series = pd.Series(values, index=observed_index, name='value')
    return series


def resample_series(series: pd.Series, rule: str) -> pd.Series:
    return series.resample(rule).ffill()


def interpolate_series(series: pd.Series, rule: str) -> pd.Series:
    return series.resample(rule).interpolate(method='linear')


def gaussian_process_fill(series: pd.Series, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series_sorted = series.sort_index()
    x = (series_sorted.index.view(int) / 1e9).to_numpy().reshape(-1, 1)
    y = series_sorted.values

    kernel = RBF(length_scale=config.gp_length_scale)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1.0)
    gp.fit(x, y)

    full_index = pd.date_range(series_sorted.index.min(), series_sorted.index.max(), freq=config.resample_rule)
    x_new = (full_index.view(int) / 1e9).reshape(-1, 1)
    y_pred, std = gp.predict(x_new, return_std=True)
    return full_index, y_pred, std


def plot_results(original: pd.Series, resampled: pd.Series, interpolated: pd.Series,
                 gp_index: np.ndarray, gp_mean: np.ndarray, gp_std: np.ndarray,
                 config: Config) -> None:
    fig, ax = setup_figure((12, 6), 150)
    apply_plot_style(ax, {'plotting': {
        'style': {'spines': {'top': False, 'right': False, 'bottom': True, 'left': True}, 'grid': False}
    }})

    ax.scatter(original.index, original.values, label='Original Irregular', color='black')
    ax.plot(resampled.index, resampled.values, label='Resampled (ffill)', color='royalblue')
    ax.plot(interpolated.index, interpolated.values, label='Interpolated (linear)', color='seagreen')
    ax.plot(gp_index, gp_mean, label='GP mean', color='tomato')
    ax.fill_between(gp_index, gp_mean - 1.96 * gp_std, gp_mean + 1.96 * gp_std,
                    color='tomato', alpha=0.2, label='GP 95% interval')

    ax.set_title('Handling Irregular Time Series')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    apply_legend(ax, {'frameon': False, 'loc': 'best'})

    output_path = config.output_dir / 'irregular_time_series.png'
    save_plot(fig, output_path)
    plt.close(fig)
    print(f"✓ Plot saved -> {output_path}")


def save_series(resampled: pd.Series, interpolated: pd.Series, gp_index: np.ndarray,
                gp_mean: np.ndarray, gp_std: np.ndarray, config: Config) -> None:
    resampled.to_frame(name='resampled').to_csv(config.output_dir / 'resampled.csv')
    interpolated.to_frame(name='interpolated').to_csv(config.output_dir / 'interpolated.csv')
    gp_df = pd.DataFrame({'timestamp': gp_index, 'gp_mean': gp_mean, 'gp_std': gp_std})
    gp_df.to_csv(config.output_dir / 'gaussian_process.csv', index=False)
    print("✓ CSV outputs saved")


def main():
    config = load_config()
    original = simulate_irregular_series(config)
    resampled = resample_series(original, config.resample_rule)
    interpolated = interpolate_series(original, config.resample_rule)
    gp_index, gp_mean, gp_std = gaussian_process_fill(original, config)

    plot_results(original, resampled, interpolated, gp_index, gp_mean, gp_std, config)
    save_series(resampled, interpolated, gp_index, gp_mean, gp_std, config)


if __name__ == "__main__":
    main()
