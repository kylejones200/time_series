#!/usr/bin/env python3
"""TimesFM experiment run on the ml-ready Net Generation dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timesfm
import yaml
from sklearn.metrics import mean_absolute_error


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
    backend: str
    per_core_batch_size: int
    context_length: int
    horizon_length: int
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
        freq=cfg["data"].get("frequency", "M"),
        history_end=pd.Timestamp(experiment["history_end"]),
        forecast_start=pd.Timestamp(experiment["forecast_start"]),
        forecast_end=pd.Timestamp(experiment["forecast_end"]),
        checkpoint=model_cfg["checkpoint"],
        backend=model_cfg.get("backend", "cpu"),
        per_core_batch_size=int(model_cfg.get("per_core_batch_size", 32)),
        context_length=int(model_cfg.get("context_length", 512)),
        horizon_length=int(model_cfg.get("horizon_length", 8)),
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


def build_model(config: Config) -> timesfm.TimesFm:
    hparams_kwargs = {
        "backend": config.backend,
        "per_core_batch_size": config.per_core_batch_size,
        "horizon_len": config.horizon_length,
    }
    if "context_length" in timesfm.TimesFmHparams.__dataclass_fields__:
        hparams_kwargs["context_length"] = config.context_length
    hparams = timesfm.TimesFmHparams(**hparams_kwargs)
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=config.checkpoint)
    return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)


def prepare_training_frame(series: pd.Series, end: pd.Timestamp) -> pd.DataFrame:
    train = series.loc[:end]
    return pd.DataFrame(
        {"unique_id": ["EIA"] * len(train), "ds": train.index, "y": train.values}
    )


def run_timesfm_forecast(
    model: timesfm.TimesFm, df: pd.DataFrame, freq: str
) -> pd.Series:
    fc = model.forecast_on_df(inputs=df, freq=freq, value_name="y", num_jobs=-1)
    fc = fc[fc["unique_id"] == df["unique_id"].iloc[0]].copy()
    fc = fc.set_index("ds").sort_index()
    for candidate in ["y_hat", "yhat", "mean", "y", "point_forecast"]:
        if candidate in fc.columns:
            return fc[candidate]
    numeric_cols = [c for c in fc.columns if pd.api.types.is_numeric_dtype(fc[c])]
    if numeric_cols:
        return fc[numeric_cols[0]]
    raise RuntimeError(
        f"No numeric forecast column found in TimesFM output: {list(fc.columns)}"
    )


def compute_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    aligned_actual, aligned_forecast = actual.align(forecast, join="inner")
    if aligned_actual.empty:
        return {}
    errors = aligned_forecast.values - aligned_actual.values
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    denom = np.where(
        aligned_actual.values == 0, np.finfo(float).eps, aligned_actual.values
    )
    mape = float(np.mean(np.abs(errors / denom)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def save_metrics(metrics: dict, config: Config) -> None:
    if not metrics:
        return
    metrics_path = config.output_dir / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.safe_dump({k: float(v) for k, v in metrics.items()}, f)
    print(f" Metrics saved -> {metrics_path}")


def plot_tufte(
    series: pd.Series,
    history_end: pd.Timestamp,
    forecast_series: pd.Series,
    actual: pd.Series,
    config: Config,
    metrics: dict,
) -> None:
    start_2024 = pd.Timestamp("2024-01-01")
    history = series.loc[start_2024:history_end]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, color="#888888", lw=1.5)
    ax.axvline(config.forecast_start, color="#666666", linestyle="--", lw=1)
    if not actual.empty:
        ax.plot(actual.index, actual.values, color="#444444", lw=1.8)
    if not forecast_series.empty:
        ax.plot(forecast_series.index, forecast_series.values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("EIA Net Generation — TimesFM forecast Jan–Aug 2025")

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
    if not forecast_series.empty:
        ax.annotate(
            "TimesFM",
            xy=(forecast_series.index[-1], forecast_series.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#000000",
        )
    if metrics:
        metrics_text = "\n".join(f"{k}: {v:.2f}" for k, v in metrics.items())
        ax.text(
            0.02,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    fig.tight_layout()
    fig.savefig(config.output_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" TimesFM plot saved -> {config.output_plot}")


def main() -> None:
    config = load_config()
    series = load_series(config)

    actual = series.loc[config.forecast_start : config.forecast_end]
    train_df = prepare_training_frame(series, config.history_end)

    model = build_model(config)
    forecast_series = run_timesfm_forecast(model, train_df, config.freq)
    forecast_slice = forecast_series.loc[config.forecast_start : config.forecast_end]

    metrics = compute_metrics(actual, forecast_slice)
    save_metrics(metrics, config)
    if metrics:
        print("TimesFM Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:,.2f}")

    plot_tufte(series, config.history_end, forecast_slice, actual, config, metrics)


if __name__ == "__main__":
    main()
