#!/usr/bin/env python3
"""Moirai forecasting template aligned with the 2025-11-08 article assets."""

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
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


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
    num_samples: int
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
        num_samples=int(model_cfg.get("num_samples", 100)),
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


def build_moirai(config: Config) -> MoiraiForecast:
    module = MoiraiModule.from_pretrained(config.checkpoint)
    forecast_model = MoiraiForecast(
        prediction_length=config.horizon,
        target_dim=1,
        context_length=config.context_length,
        module=module,
        num_samples=config.num_samples,
    )
    return forecast_model


def generate_forecast(
    model: MoiraiForecast, train_series: pd.Series, config: Config
) -> np.ndarray:
    dataset = ListDataset(
        [
            {
                "target": train_series.values.astype(np.float32),
                "start": train_series.index[0],
            }
        ],
        freq=config.freq,
    )
    predictor = model.create_predictor(batch_size=1, device="cpu")
    prediction_iterator = predictor.predict(dataset)

    forecast_array = None
    for forecast in prediction_iterator:
        if hasattr(forecast, "mean"):
            forecast_array = forecast.mean
        else:
            forecast_array = forecast.samples.mean(axis=0)
        break

    if forecast_array is None:
        raise RuntimeError("Moirai did not return any forecasts.")

    return np.asarray(forecast_array).reshape(-1)


def plot_tufte(
    series: pd.Series,
    history_end: pd.Timestamp,
    forecast_values: np.ndarray,
    actual: pd.Series,
    config: Config,
) -> None:
    start_2024 = pd.Timestamp("2024-01-01")
    history = series.loc[start_2024:history_end]
    forecast_index = pd.period_range(
        config.forecast_start, config.forecast_end, freq="M"
    ).to_timestamp()
    forecast_series = pd.Series(
        forecast_values[: len(forecast_index)], index=forecast_index
    )

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
    ax.set_title("EIA Net Generation — Moirai forecast Jan–Aug 2025")

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
        "Moirai",
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
    print(f"✓ Moirai plot saved -> {config.output_plot}")


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    config = load_config()
    series = load_series(config)

    train_series = series.loc[: config.history_end]
    actual = series.loc[config.forecast_start : config.forecast_end]

    model = build_moirai(config)
    forecast_values = generate_forecast(model, train_series, config)

    plot_tufte(series, config.history_end, forecast_values, actual, config)


if __name__ == "__main__":
    main()
