#!/usr/bin/env python3
"""Darts project template aligned with the 2025-11-08 article assets."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from darts import TimeSeries
from darts.models import ARIMA, ExponentialSmoothing, NaiveSeasonal, Theta

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ts_utils import load_ts_data  # noqa: E402

np.random.seed(42)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)


MODEL_REGISTRY = {
    "ARIMA": ARIMA,
    "Theta": Theta,
    "ExponentialSmoothing": ExponentialSmoothing,
    "NaiveSeasonal": NaiveSeasonal,
}


@dataclass
class EvalResult:
    model_name: str
    mean_mae: float
    y_true: Optional[TimeSeries]
    y_pred: Optional[TimeSeries]


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_series(config: dict) -> pd.Series:
    data_cfg = config["data"]
    data_path = repo_root() / "data" / data_cfg["input_file"]
    series = load_ts_data(
        data_path,
        date_col=data_cfg["date_col"],
        value_col=data_cfg["value_col"],
    )
    freq = data_cfg.get("freq")
    if freq:
        series = series.asfreq(freq)
    return series.astype(float).dropna()


def make_model_factory(model_cfg: dict):
    model_cls = MODEL_REGISTRY[model_cfg["type"]]
    params = model_cfg.get("params", {})
    return lambda: model_cls(**params)


def rolling_origin_eval(
    ts: TimeSeries, model_cfg: dict, horizon: int, n_splits: int
) -> EvalResult:
    model_factory = make_model_factory(model_cfg)
    indices = np.arange(len(ts))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    maes: List[float] = []
    last_true: Optional[TimeSeries] = None
    last_pred: Optional[TimeSeries] = None

    for train_idx, _ in splitter.split(indices):
        cutoff = ts.time_index[train_idx[-1]]
        train_ts, future_ts = ts.split_after(cutoff)
        if len(future_ts) == 0:
            continue
        window_end = min(horizon, len(future_ts)) - 1
        test_ts = future_ts.drop_after(future_ts.time_index[window_end])

        model = model_factory()
        model.fit(train_ts)
        forecast = model.predict(len(test_ts))

        maes.append(
            mean_absolute_error(
                test_ts.values().ravel(), forecast.values().ravel()
            )
        )
        last_true, last_pred = test_ts, forecast

    mean_mae = float(np.mean(maes)) if maes else float("nan")
    return EvalResult(model_cfg["name"], mean_mae, last_true, last_pred)


def evaluate_group(
    ts: TimeSeries, group_models: Iterable[dict], eval_cfg: dict
) -> List[EvalResult]:
    return [
        rolling_origin_eval(ts, model_cfg, eval_cfg["horizon"], eval_cfg["n_splits"])
        for model_cfg in group_models
    ]


def maybe_load_tbats(tbats_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not tbats_path:
        return None
    path = repo_root() / tbats_path
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


def to_series(ts: Optional[TimeSeries]) -> Optional[pd.Series]:
    return None if ts is None else ts.pd_series()


def plot_tufte_view(
    series: pd.Series,
    results: List[EvalResult],
    eval_cfg: dict,
    output_dir: Path,
) -> None:
    history = series.loc[eval_cfg["history_start"] : eval_cfg["history_end"]]
    actual = series.loc[eval_cfg["actual_start"] : eval_cfg["actual_end"]]
    tbats_df = maybe_load_tbats(eval_cfg.get("tbats_csv"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, color="#888888", lw=1.5)
    ax.axvline(pd.Timestamp(eval_cfg["actual_start"]), color="#666666", linestyle="--", lw=1)

    if not actual.empty:
        ax.plot(actual.index, actual.values, color="#444444", lw=1.8)

    end_labels: List[tuple] = []
    for result in results:
        forecast_series = to_series(result.y_pred)
        if forecast_series is None:
            continue
        mask = (forecast_series.index >= eval_cfg["actual_start"]) & (
            forecast_series.index <= eval_cfg["actual_end"]
        )
        if mask.any():
            filtered = forecast_series.loc[mask]
            ax.plot(filtered.index, filtered.values, color="#000000", lw=2.0, alpha=0.85)
            end_labels.append((filtered.index[-1], filtered.values[-1], result.model_name))

    if tbats_df is not None and not tbats_df.empty:
        tbats_filtered = tbats_df.loc[
            (tbats_df["date"] >= eval_cfg["actual_start"])
            & (tbats_df["date"] <= eval_cfg["actual_end"])
        ]
        if not tbats_filtered.empty:
            ax.plot(tbats_filtered["date"], tbats_filtered["pred"], color="#000000", lw=1.6, alpha=0.6)
            end_labels.append(
                (
                    tbats_filtered["date"].iloc[-1],
                    tbats_filtered["pred"].iloc[-1],
                    "TBATS",
                )
            )

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("EIA Net Generation — Darts last-fold forecasts Jan–Aug 2025")

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
    for x_coord, y_coord, label in end_labels:
        ax.annotate(
            label,
            xy=(x_coord, y_coord),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#000000",
        )

    output_path = output_dir / eval_cfg["output_plot"]
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Tufte plot saved -> {output_path}")


def plot_overview(series: TimeSeries, results: List[EvalResult], output_cfg: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    series_df = series.pd_dataframe()
    ax.plot(series_df.index, series_df.values, label="History", alpha=0.6, color="#444444")

    for result in results:
        forecast_series = to_series(result.y_pred)
        if forecast_series is None:
            continue
        ax.plot(
            forecast_series.index,
            forecast_series.values,
            label=f"{result.model_name} last fold",
        )

    ax.legend(frameon=False)
    ax.set_title("Darts overview — last fold forecasts")
    ax.set_xlabel("")
    ax.set_ylabel("Value")

    output_path = output_dir / output_cfg["output_plot"]
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Overview plot saved -> {output_path}")


def save_predictions(results: List[EvalResult], output_dir: Path, filename: str) -> None:
    rows: List[pd.DataFrame] = []
    for result in results:
        y_true = to_series(result.y_true)
        y_pred = to_series(result.y_pred)
        if y_true is None or y_pred is None:
            continue
        aligned = pd.DataFrame(
            {
                "model": result.model_name,
                "date": y_true.index,
                "true": y_true.values,
                "pred": y_pred.reindex(y_true.index).values,
            }
        )
        rows.append(aligned.dropna())

    if not rows:
        return
    output_path = output_dir / filename
    pd.concat(rows).to_csv(output_path, index=False)
    print(f"✓ Predictions saved -> {output_path}")


def print_metrics(results: Dict[str, List[EvalResult]]) -> None:
    for group_name, group_results in results.items():
        print(f"\n=== {group_name} metrics ===")
        for result in group_results:
            print(f"{result.model_name}: MAE={result.mean_mae:.3f}")


def main() -> None:
    config = load_config()
    series = load_series(config)
    ts = TimeSeries.from_series(series)
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    tufte_results = evaluate_group(ts, config["models"]["tufte_last_fold"], config["evaluations"]["tufte_last_fold"])
    overview_results = evaluate_group(ts, config["models"]["overview_last_fold"], config["evaluations"]["overview_last_fold"])

    plot_tufte_view(series, tufte_results, config["evaluations"]["tufte_last_fold"], output_dir)
    plot_overview(ts, overview_results, config["evaluations"]["overview_last_fold"], output_dir)
    save_predictions(tufte_results, output_dir, config["output"]["save_predictions_csv"])

    print_metrics(
        {
            "Tufte last fold": tufte_results,
            "Overview last fold": overview_results,
        }
    )
    print("✓ Darts evaluation complete")


if __name__ == "__main__":
    main()

