#!/usr/bin/env python3
"""Granite TTM forecasting template aligned with the 2025-11-08 article assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import signalplot
import numpy as np
import pandas as pd
import torch
import yaml
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    history_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    checkpoint: str
    context_length: int
    horizon: int
    output_dir: Path
    output_plot: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg["data"]["input_file"]
    output_dir = Path(__file__).parent / cfg["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    experiment = cfg["experiment"]
    model_cfg = cfg["model"]

    return Config(
        data_path=data_path,
        date_col=cfg["data"]["date_col"],
        value_col=cfg["data"]["value_col"],
        freq=cfg["data"].get("freq", "MS"),
        history_end=pd.Timestamp(experiment["history_end"]),
        forecast_start=pd.Timestamp(experiment["forecast_start"]),
        forecast_end=pd.Timestamp(experiment["forecast_end"]),
        checkpoint=model_cfg["checkpoint"],
        context_length=int(model_cfg.get("context_length", 512)),
        horizon=int(model_cfg.get("horizon", 8)),
        output_dir=output_dir,
        output_plot=output_dir / cfg["output"]["tufte_plot"],
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


def build_input_context(
    series: pd.Series, end_timestamp: pd.Timestamp, context_length: int
) -> torch.Tensor:
    context_values = series.loc[:end_timestamp].values.astype(np.float32)
    if len(context_values) >= context_length:
        context_values = context_values[-context_length:]
    else:
        pad = np.zeros(context_length - len(context_values), dtype=np.float32)
        context_values = np.concatenate([pad, context_values])
    return torch.tensor(context_values, dtype=torch.float32).view(1, context_length, 1)


def generate_granite_forecast(
    series: pd.Series, config: Config
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    model = get_model(
        config.checkpoint,
        context_length=config.context_length,
        prediction_length=config.horizon,
        freq="W",
    )

    if hasattr(model, "prediction_filter_length"):
        model.prediction_filter_length = config.horizon

    context_tensor = build_input_context(
        series, config.history_end, config.context_length
    )

    preprocessor = TimeSeriesPreprocessor(
        freq="W",
        context_length=config.context_length,
        prediction_length=config.horizon,
    )
    freq_token_id = preprocessor.get_frequency_token("W")
    freq_token = torch.tensor([freq_token_id], dtype=torch.long)

    with torch.no_grad():
        output = model(context_tensor, freq_token=freq_token)

    if hasattr(output, "prediction_outputs"):
        forecast = output.prediction_outputs
    else:
        try:
            forecast = output[0]
        except Exception:
            forecast = output

    forecast = np.asarray(forecast).reshape(-1)[: config.horizon]
    forecast_index = pd.period_range(
        config.forecast_start, config.forecast_end, freq="M"
    ).to_timestamp()
    return forecast_index, forecast


def plot_tufte(
    series: pd.Series,
    history_end: pd.Timestamp,
    forecast_index: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    config: Config,
) -> None:
    start_2024 = pd.Timestamp("2024-01-01")
    history = series.loc[start_2024:history_end]
    actual = series.loc[config.forecast_start : config.forecast_end]
    forecast_series = pd.Series(forecast_values, index=forecast_index)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, color="#888888", lw=1.5)
    ax.axvline(config.forecast_start, color="#666666", linestyle="--", lw=1)
    if not actual.empty:
        ax.plot(actual.index, actual.values, color="#444444", lw=1.8)
    ax.plot(forecast_series.index, forecast_series.values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

# Apply SignalPlot's clean defaults
signalplot.apply()

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("EIA Net Generation — Granite TTM forecast Jan–Aug 2025")

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
        "Granite TTM",
        xy=(forecast_series.index[-1], forecast_series.values[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        va="center",
        ha="left",
        color="#000000",
    )

    fig.tight_layout()
    fig.savefig(config.output_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Granite TTM plot saved -> {config.output_plot}")


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    config = load_config()
    series = load_series(config)

    forecast_index, forecast_values = generate_granite_forecast(series, config)
    plot_tufte(series, config.history_end, forecast_index, forecast_values, config)


if __name__ == "__main__":
    main()
