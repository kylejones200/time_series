#!/usr/bin/env python3
"""Darts project template using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from darts import TimeSeries
from darts.models import ARIMA, ExponentialSmoothing, NaiveSeasonal, Theta

np.random.seed(42)


MODEL_REGISTRY = {
    "ARIMA": ARIMA,
    "Theta": Theta,
    "ExponentialSmoothing": ExponentialSmoothing,
    "NaiveSeasonal": NaiveSeasonal,
}


@dataclass
class EvalResult:
    """Evaluation result dataclass."""
    model_name: str
    mean_mae: float
    y_true: Optional[TimeSeries]
    y_pred: Optional[TimeSeries]


def load_series(config: dict) -> pd.Series:
    """Load time series using consolidated loader."""
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value")
    )
    
    freq = config["data"].get("freq")
    if freq:
        series = series.asfreq(freq)
    
    return series.astype(float).dropna()


def make_model_factory(model_cfg: dict):
    """Create model factory function."""
    model_cls = MODEL_REGISTRY[model_cfg["type"]]
    params = model_cfg.get("params", {})
    return lambda: model_cls(**params)


def rolling_origin_eval(
    ts: TimeSeries, model_cfg: dict, horizon: int, n_splits: int
) -> EvalResult:
    """Rolling origin evaluation for Darts models."""
    model_factory = make_model_factory(model_cfg)
    values = ts.values().flatten()
    idx = np.arange(len(values))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    last_true = None
    last_pred = None
    
    for train_idx, _ in splitter.split(idx):
        end_idx = train_idx[-1]
        train_ts = ts[: end_idx + 1]
        future_ts = ts[end_idx + 1 : end_idx + 1 + horizon]
        
        if len(future_ts) < horizon:
            continue
        
        model = model_factory()
        model.fit(train_ts)
        forecast = model.predict(horizon)
        
        mae_val = mean_absolute_error(future_ts.values().flatten(), forecast.values().flatten())
        maes.append(mae_val)
        
        last_true = future_ts
        last_pred = forecast
    
    mean_mae = float(np.mean(maes)) if maes else float("nan")
    return EvalResult(
        model_name=model_cfg["type"],
        mean_mae=mean_mae,
        y_true=last_true,
        y_pred=last_pred,
    )


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load series
    series = load_series(config)
    print(f"Loaded {len(series)} data points")
    
    # Convert to Darts TimeSeries
    ts = TimeSeries.from_series(series)
    
    # Evaluate models
    horizon = config["model"]["horizon"]
    n_splits = config["model"]["n_splits"]
    models_config = config["model"]["models"]
    
    results = []
    for model_cfg in models_config:
        print(f"\nEvaluating {model_cfg['type']}...")
        result = rolling_origin_eval(ts, model_cfg, horizon, n_splits)
        results.append(result)
        print(f"  Mean MAE: {result.mean_mae:.4f}")
    
    # Find best model
    best_result = min(results, key=lambda r: r.mean_mae)
    print(f"\nBest model: {best_result.model_name} (MAE: {best_result.mean_mae:.4f})")
    
    # Create visualization
    if best_result.y_true is not None and best_result.y_pred is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot history
        history = ts[-100:] if len(ts) > 100 else ts
        ax.plot(history.time_index, history.values().flatten(), "k-", lw=1.5, label="History", alpha=0.8)
        
        # Plot actual and forecast
        ax.plot(best_result.y_true.time_index, best_result.y_true.values().flatten(), "b-", lw=1.8, label="Actual", alpha=0.8)
        ax.plot(best_result.y_pred.time_index, best_result.y_pred.values().flatten(), "r--", lw=2.0, label=f"{best_result.model_name} Forecast", alpha=0.8)
        
        ax.set_title(f"Best Model: {best_result.model_name} (MAE: {best_result.mean_mae:.4f})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "darts_forecast.png", dpi=300)
        print(f"\nPlot saved to: {output_dir / 'darts_forecast.png'}")
        plt.close(fig)
    
    print("\n Darts analysis complete")


if __name__ == "__main__":
    main()
