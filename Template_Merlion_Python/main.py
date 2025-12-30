#!/usr/bin/env python3
"""
Merlion: Time Series Forecasting and Anomaly Detection
Unified framework for time series forecasting and anomaly detection.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from merlion.models.forecast.prophet import ProphetForecaster
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.anomaly.isolation_forest import IsolationForest
from merlion.models.anomaly.auto_encoder import AutoEncoder
from merlion.utils import TimeSeries
from merlion.evaluate.forecast import ForecastEvaluator
from merlion.evaluate.anomaly import TSADEvaluator
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Apply SignalPlot's clean defaults
signalplot.apply()


def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj



ts_utils = repo_import("utils.ts_utils")
load_ts_data = ts_utils.load_ts_data
ensure_datetime_index = ts_utils.ensure_datetime_index
split_ts = ts_utils.split_ts


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(config):
    """Load time series data."""
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / "data" / config["data"]["input_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["value_col"],
    )
    df = ensure_datetime_index(df, time_col=config["data"]["date_col"])
    return df


def create_forecaster(config):
    """Create Merlion forecaster based on config."""
    forecaster_map = {
        "Prophet": lambda: ProphetForecaster(
            **config["model"].get("forecaster_params", {})
        ),
        "ARIMA": lambda: Arima(
            ArimaConfig(
                order=tuple(config["model"].get("arima_order", [2, 1, 2])),
                max_forecast_steps=config["model"].get("forecast_horizon", 12),
            )
        ),
    }

    return forecaster_map.get(
        config["model"]["forecaster_type"], forecaster_map["Prophet"]
    )()


def create_anomaly_detector(config):
    """Create Merlion anomaly detector based on config."""
    detector_map = {
        "IsolationForest": lambda: IsolationForest(
            **config["model"].get("detector_params", {})
        ),
        "AutoEncoder": lambda: AutoEncoder(
            **config["model"].get("detector_params", {})
        ),
    }

    return detector_map.get(
        config["model"]["detector_type"], detector_map["IsolationForest"]
    )()


def fit_and_forecast(forecaster, data, config):
    """Fit forecaster and generate predictions."""
    train_df, test_df = split_ts(data, test_size=config["data"]["test_size"])

    train_data = TimeSeries.from_pd(train_df)
    test_data = TimeSeries.from_pd(test_df)

    forecaster.train(train_data)
    predictions, _ = forecaster.forecast(time_stamps=test_data.time_stamps)

    test_values = test_data.to_pd().values.flatten()
    pred_values = predictions.to_pd().values.flatten()

    mae = mean_absolute_error(test_values, pred_values)
    rmse = np.sqrt(mean_squared_error(test_values, pred_values))

    print(f"\nForecast Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return train_data, test_data, predictions


def detect_anomalies(detector, data, config):
    """Detect anomalies in time series."""
    ts_data = TimeSeries.from_pd(data)
    detector.train(ts_data)
    predictions = detector.get_anomaly_label(ts_data)

    return predictions


def create_forecast_visualization(train_data, test_data, predictions, config):
    """Generate forecast visualization."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=tuple(config["plotting"]["figure_size"]))
    
    train_df = train_data.to_pd()
    test_df = test_data.to_pd()
    pred_df = predictions.to_pd()

    ax.plot(
        train_df.index,
        train_df.values.flatten(),
        "k-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label="Train",
    )

    ax.plot(
        test_df.index,
        test_df.values.flatten(),
        "g-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label="Test",
    )

    ax.plot(
        pred_df.index,
        pred_df.values.flatten(),
        "r--",
        linewidth=config["plotting"]["linewidth"],
        label="Forecast",
    )

    ax.set_title(config["plot_titles"]["merlion_forecast"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    output_path = output_dir / "merlion_forecast.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def create_anomaly_visualization(data, predictions, config):
    """Generate anomaly detection visualization."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=tuple(config["plotting"]["figure_size"]))
    
    data_df = data.to_pd() if hasattr(data, "to_pd") else data
    pred_df = predictions.to_pd() if hasattr(predictions, "to_pd") else predictions

    anomaly_labels = pred_df.values.flatten()
    anomalies = data_df[anomaly_labels == 1]
    normal = data_df[anomaly_labels == 0]

    ax.plot(
        normal.index,
        normal.values.flatten(),
        "k-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label="Normal",
    )

    ax.scatter(
        anomalies.index,
        anomalies.values.flatten(),
        c="r",
        s=config["plotting"]["markersize"] * 20,
        label="Anomalies",
        zorder=5,
    )

    ax.set_title(config["plot_titles"]["merlion_anomalies"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    output_path = output_dir / "merlion_anomalies.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    print(f"\nAnomaly Detection Results:")
    print(f"Total anomalies detected: {len(anomalies)}")
    print(f"Anomaly rate: {len(anomalies) / len(data_df) * 100:.2f}%")


def main():
    """Main execution function."""
    config = load_config()
    data = load_data(config)

    task_map = {
        "forecast": lambda: (
            create_forecaster(config),
            lambda f: fit_and_forecast(f, data, config),
            lambda t, te, p: create_forecast_visualization(t, te, p, config),
        ),
        "anomaly": lambda: (
            create_anomaly_detector(config),
            lambda d: detect_anomalies(d, data, config),
            lambda d, p: create_anomaly_visualization(d, p, config),
        ),
    }

    task = config["model"]["task"]
    model, fit_func, viz_func = task_map[task]()

    results = fit_func(model)
    viz_func(data, results) if task == "anomaly" else viz_func(*results)

    print(f"✓ Merlion {task} complete")


if __name__ == "__main__":
    main()
