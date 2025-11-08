#!/usr/bin/env python3
"""ERCOT load preprocessing and minimalist visualization."""

from __future__ import annotations

from pathlib import Path

import yaml
import pandas as pd

from utils.ts_utils import ensure_datetime_index
from utils.plotting_utils import setup_figure, apply_plot_style, save_plot, apply_legend


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_ercot_data(data_path: Path, timestamp_col: str, load_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    return df[load_cols]


def filter_and_resample(df: pd.DataFrame, min_load: float, resample_freq: str) -> pd.DataFrame:
    df = ensure_datetime_index(df)
    filtered = df[df.iloc[:, 0] >= min_load]
    return filtered.resample(resample_freq).ffill(limit=1)


def compute_differences(df: pd.DataFrame) -> pd.Series:
    return df.iloc[:, 0].diff().dropna()


def plot_series(series: pd.Series, config: dict, title: str, output_name: str) -> None:
    plotting_cfg = {'plotting': config['plotting']}
    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, plotting_cfg)
    ax.plot(series.index, series.values, color=config['plotting']['colors'][0], linewidth=config['plotting']['linewidth'])
    ax.set_title(title)
    ax.set_xlabel(config['axes']['x_label'])
    ax.set_ylabel(config['axes']['y_label'])
    apply_legend(ax, config['plotting']['legend'])
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    save_plot(fig, outputs_dir / output_name, config['output'])


def main() -> None:
    project_root = Path(__file__).parent.parent.parent
    config = load_config(Path(__file__).parent / "config.yaml")

    data_path = project_root / "data" / config['data']['input_file']
    df = load_ercot_data(data_path, config['data']['timestamp_col'], config['data']['load_columns'])

    filtered = filter_and_resample(df, config['processing']['min_load'], config['processing']['resample_frequency'])
    filtered_output = Path(__file__).parent / "outputs" / config['output']['filtered_csv']
    filtered.to_csv(filtered_output)

    diff_series = compute_differences(filtered)

    plot_series(filtered.iloc[:, 0], config, config['plot_titles']['filtered_load'], "filtered_load.png")
    plot_series(diff_series, config, config['plot_titles']['load_difference'], "load_difference.png")

    print("✓ ERCOT preprocessing complete")
    print(f"Filtered series saved to: {filtered_output}")


if __name__ == "__main__":
    main()

