#!/usr/bin/env python3
"""Amazon Chronos forecasting template aligned with the 2025-11-08 article assets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from chronos import ChronosPipeline
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error


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

np.random.seed(42)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    resample_rule: Optional[str]
    frequency: Optional[str]
    context_length: Optional[int]
    prediction_length: int
    model_name: str
    device_map: str
    torch_dtype: str
    num_samples: int
    output_dir: Path
    tufte_cfg: dict
    forecast_plot_cfg: dict


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg["data"]["input_file"]
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    plotting_cfg = cfg.get("plotting", {})
    tufte_cfg = plotting_cfg.get("tufte", {})
    forecast_cfg = plotting_cfg.get("forecast", {})

    return Config(
        data_path=data_path,
        date_col=cfg["data"]["date_col"],
        value_col=cfg["data"]["value_col"],
        resample_rule=cfg["data"].get("resample_rule"),
        frequency=cfg["data"].get("frequency"),
        context_length=cfg["model"].get("context_length"),
        prediction_length=cfg["model"]["prediction_length"],
        model_name=cfg["model"]["huggingface_model"],
        device_map=cfg["model"].get("device_map", "cpu"),
        torch_dtype=cfg["model"].get("torch_dtype", "float32"),
        num_samples=cfg["model"].get("num_samples", 20),
        output_dir=output_dir,
        tufte_cfg=tufte_cfg,
        forecast_plot_cfg=forecast_cfg,
    )


def load_series(config: Config) -> pd.Series:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")

    df = pd.read_csv(config.data_path)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not present in CSV")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.value_col])
    df = df.sort_values(config.date_col)
    df = df.set_index(config.date_col)

    if config.resample_rule:
        df = df.resample(config.resample_rule).mean()

    series = pd.to_numeric(df[config.value_col], errors="coerce").dropna()

    if config.frequency:
        series = series.asfreq(config.frequency)

    if len(series) <= config.prediction_length:
        raise ValueError("Time series length must exceed prediction length.")

    return series


def prepare_tensors(series: pd.Series, config: Config) -> tuple[torch.Tensor, torch.Tensor, pd.Index]:
    pred_len = config.prediction_length
    context_len = config.context_length or len(series) - pred_len
    context_len = min(context_len, len(series) - pred_len)

    train_values = series.iloc[-(pred_len + context_len) : -pred_len].values
    test_values = series.iloc[-pred_len:].values
    test_index = series.index[-pred_len:]

    dtype_attr = getattr(torch, config.torch_dtype, None)
    if dtype_attr is None:
        raise ValueError(f"Unsupported torch dtype: {config.torch_dtype}")

    context_tensor = torch.tensor(train_values, dtype=dtype_attr)
    if context_tensor.ndim == 1:
        context_tensor = context_tensor.unsqueeze(0)

    return context_tensor, torch.tensor(test_values, dtype=dtype_attr), test_index


def load_pipeline(config: Config) -> ChronosPipeline:
    kwargs = {
        "device_map": config.device_map,
        "torch_dtype": getattr(torch, config.torch_dtype, torch.float32),
    }
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        kwargs["use_auth_token"] = hf_token

    pipeline = ChronosPipeline.from_pretrained(config.model_name, **kwargs)
    pipeline.eval()
    return pipeline


def generate_forecast(pipeline: ChronosPipeline, context: torch.Tensor, config: Config) -> np.ndarray:
    with torch.no_grad():
        forecast = pipeline.predict(context, config.prediction_length, num_samples=config.num_samples)
    return forecast.cpu().numpy()


def compute_stats(forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, median, high = np.quantile(forecast[0], [0.1, 0.5, 0.9], axis=0)
    return low, median, high


def compute_metrics(true: torch.Tensor, median: np.ndarray) -> dict:
    true_np = true.cpu().numpy()
    mae = mean_absolute_error(true_np, median)
    rmse = np.sqrt(mean_squared_error(true_np, median))
    mape = np.mean(np.abs((true_np - median) / true_np)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def plot_forecast(
    series: pd.Series,
    forecast_index: pd.Index,
    median: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    metrics: dict,
    config: Config,
) -> None:
    history_window = 7 * config.prediction_length
    history = series.iloc[-history_window:]

    fig, ax = setup_figure((14, 6), 150)
    apply_plot_style(
        ax,
        {
            "plotting": {
                "style": {
                    "spines": {"top": False, "right": False, "bottom": True, "left": True},
                    "grid": False,
                }
            }
        },
    )

    ax.plot(history.index, history.values, color="royalblue", linewidth=2, label="History")
    ax.plot(forecast_index, median, color="tomato", linewidth=2, label="Chronos Forecast")
    ax.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% interval")

    plot_title = config.forecast_plot_cfg.get("title", "Chronos Forecast")
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(config.value_col)

    metrics_text = "\n".join(f"{k}: {v:.3f}" for k, v in metrics.items())
    ax.text(
        0.02,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    apply_legend(ax, {"frameon": False, "loc": "best"})

    output_name = config.forecast_plot_cfg.get("output_plot", "chronos_forecast.png")
    output_path = config.output_dir / output_name
    save_plot(fig, output_path)
    plt.close(fig)
    print(f"✓ Forecast plot saved -> {output_path}")


def plot_tufte_style(series: pd.Series, forecast_index: pd.Index, median: np.ndarray, config: Config) -> None:
    if not config.tufte_cfg:
        return

    tufte_cfg = config.tufte_cfg
    history = series.loc[tufte_cfg["history_start"] : tufte_cfg["history_end"]]
    actual = series.loc[tufte_cfg["actual_start"] : tufte_cfg["actual_end"]]
    forecast_series = pd.Series(median, index=forecast_index)
    forecast_filtered = forecast_series.loc[tufte_cfg["actual_start"] : tufte_cfg["actual_end"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, color="#888888", lw=1.5)
    ax.axvline(pd.Timestamp(tufte_cfg["actual_start"]), color="#666666", linestyle="--", lw=1)
    if not actual.empty:
        ax.plot(actual.index, actual.values, color="#444444", lw=1.8)
    if not forecast_filtered.empty:
        ax.plot(forecast_filtered.index, forecast_filtered.values, color="#000000", lw=2.0)

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("EIA Net Generation — Chronos forecast Jan–Aug 2025")

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
    if not forecast_filtered.empty:
        ax.annotate(
            "Chronos",
            xy=(forecast_filtered.index[-1], forecast_filtered.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#000000",
        )

    output_name = tufte_cfg.get("output_plot", "eia_chronos_last_fold.png")
    output_path = config.output_dir / output_name
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Tufte-style plot saved -> {output_path}")


def save_forecast(
    forecast_index: pd.Index,
    median: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    metrics: dict,
    config: Config,
) -> None:
    forecast_df = pd.DataFrame(
        {
            "date": forecast_index,
            "median": median,
            "lower_p10": low,
            "upper_p90": high,
        }
    )
    forecast_df.to_csv(config.output_dir / "chronos_forecast.csv", index=False)
    metrics_path = config.output_dir / "chronos_metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.safe_dump(metrics, f)
    print(f"✓ Forecast data saved -> {forecast_df.shape}")
    print(f"✓ Metrics saved -> {metrics_path}")


def main():
    config = load_config()
    series = load_series(config)
    context, true_values, forecast_index = prepare_tensors(series, config)

    pipeline = load_pipeline(config)
    forecast = generate_forecast(pipeline, context, config)
    low, median, high = compute_stats(forecast)

    metrics = compute_metrics(true_values, median)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    plot_forecast(series, forecast_index, median, low, high, metrics, config)
    plot_tufte_style(series, forecast_index, median, config)
    save_forecast(forecast_index, median, low, high, metrics, config)


if __name__ == "__main__":
    main()
