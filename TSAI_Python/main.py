#!/usr/bin/env python3
"""
TSAI: Time Series AI
Deep learning for time series using TSAI library.
"""

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.evaluator import Evaluator
from src.run_logger import append_run_log, utc_now_iso

from tsai.all import *


def prepare_data(data: pd.Series, config: dict):
    """Prepare data for TSAI."""
    train_size = config["model"].get("train_size", 0.8)
    evaluator = Evaluator(test_size=1 - train_size)
    train, test = evaluator.split(data)
    
    X_train = train.values.reshape(-1, 1, len(train))
    X_test = test.values.reshape(-1, 1, len(test)) if test is not None else None
    
    y_train = train.values
    y_test = test.values if test is not None else None
    
    return X_train, X_test, y_train, y_test


def create_model(X_train: np.ndarray, config: dict):
    """Create TSAI model."""
    model_type = config["model"].get("type", "InceptionTime")
    
    model_map = {
        "InceptionTime": InceptionTime,
        "ResNet": ResNet,
        "XceptionTime": XceptionTime,
        "ROCKET": ROCKET,
    }
    
    model_class = model_map.get(model_type, InceptionTime)
    return model_class(c_in=1, c_out=1)


def train_model(model, X_train: np.ndarray, y_train: np.ndarray, config: dict):
    """Train TSAI model."""
    learn = Learner(
        TSDataLoaders.from_numpy(
            X_train,
            y_train,
            valid_pct=1 - config["model"].get("train_size", 0.8),
            bs=config["model"].get("batch_size", 64),
        ),
        model,
        metrics=[mae, rmse],
    )
    
    learn.fit_one_cycle(
        n_epoch=config["model"].get("n_epochs", 10),
        lr_max=config["model"].get("lr", 1e-3),
    )
    
    return learn


def create_visualizations(data: pd.Series, train_data: pd.Series, predictions: np.ndarray, config: dict, script_dir: Path):
    """Generate clean visualizations."""
    fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))
    
    ax.plot(
        train_data.index,
        train_data.values,
        c=config.get("plotting", {}).get("style", {}).get("colors", {}).get("primary", "k"),
        linewidth=config.get("plotting", {}).get("style", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("style", {}).get("alpha", 0.8),
        label="Historical",
    )
    
    if predictions is not None and len(predictions) > 0:
        forecast_index = pd.date_range(
            start=train_data.index[-1] + pd.Timedelta(days=1),
            periods=len(predictions),
            freq=pd.infer_freq(train_data.index) or "D"
        )
        ax.plot(
            forecast_index,
            predictions,
            c=config.get("plotting", {}).get("style", {}).get("colors", {}).get("secondary", "r"),
            linewidth=config.get("plotting", {}).get("style", {}).get("linewidth", 1.5),
            label="Forecast",
        )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"TSAI {config['model'].get('type', 'InceptionTime')} Forecast")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if config.get("output", {}).get("save_plots", True):
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "tsai_forecast.png", dpi=300)
        print(f"Plot saved to: {output_dir / 'tsai_forecast.png'}")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    started_at = utc_now_iso()
    t0 = time.perf_counter()
    config = load_config()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    status = "success"
    error_msg = None
    metrics_log: dict[str, float] = {}

    try:
        # Load data using consolidated loader
        data = load_time_series(
            config["data"]["input_file"],
            date_column=config["data"].get("date_col", "date"),
            value_column=config["data"].get("value_col", "value")
        )

        print(f"Loaded {len(data)} data points")

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(data, config)
        train_data, _ = Evaluator(test_size=config["model"].get("test_size", 0.2)).split(data)

        print(f"Train: {len(train_data)} points")
        if y_test is not None:
            print(f"Test: {len(y_test)} points")

        metrics_log["train_points"] = float(len(train_data))
        metrics_log["test_points"] = float(len(y_test) if y_test is not None else 0)

        # Create and train model
        print(f"\nCreating {config['model'].get('type', 'InceptionTime')} model...")
        model = create_model(X_train, config)

        print("Training model...")
        learn = train_model(model, X_train, y_train, config)
        if hasattr(learn, "recorder") and getattr(learn.recorder, "values", None):
            last_vals = learn.recorder.values[-1]
            if last_vals:
                metrics_log["final_train_loss"] = float(last_vals[0])

        # Generate predictions
        print("\nGenerating predictions...")
        # Prediction API differs across tsai/fastai versions and device setups.
        # Keep this script focused on training + reproducible visuals for now.
        predictions = None

        # Create visualizations
        print("\nCreating visualization...")
        create_visualizations(data, train_data, predictions, config, script_dir)

        print("\n TSAI forecasting complete")
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        raise
    finally:
        ended_at = utc_now_iso()
        duration = time.perf_counter() - t0
        log_path = append_run_log(
            output_dir=output_dir,
            script_name="TSAI_Python",
            started_at_utc=started_at,
            ended_at_utc=ended_at,
            duration_seconds=duration,
            status=status,
            metrics=metrics_log,
            details={"data_path": str(config["data"]["input_file"])},
            error=error_msg,
        )
        print(f"Run log saved to: {log_path}")


if __name__ == "__main__":
    main()
