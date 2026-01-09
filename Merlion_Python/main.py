#!/usr/bin/env python3
"""
Merlion: Time Series Forecasting and Anomaly Detection
Unified framework for time series forecasting and anomaly detection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.evaluator import Evaluator

from merlion.models.forecast.prophet import ProphetForecaster
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.anomaly.isolation_forest import IsolationForest
from merlion.models.anomaly.auto_encoder import AutoEncoder
from merlion.utils import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_forecaster(config: dict):
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


def create_anomaly_detector(config: dict):
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


def fit_and_forecast(forecaster, data: pd.DataFrame, config: dict, script_dir: Path):
    """Fit forecaster and generate predictions."""
    evaluator = Evaluator(test_size=config["data"]["test_size"])
    series = data[config["data"]["value_col"]]
    train, test = evaluator.split(series)
    
    train_df = pd.DataFrame({config["data"]["value_col"]: train})
    test_df = pd.DataFrame({config["data"]["value_col"]: test})
    
    train_data = TimeSeries.from_pd(train_df)
    test_data = TimeSeries.from_pd(test_df)
    
    forecaster.train(train_data)
    predictions, _ = forecaster.forecast(time_stamps=test_data.time_stamps)
    
    test_values = test_data.to_pd().values.flatten()
    pred_values = predictions.to_pd().values.flatten()
    
    mae_val = mean_absolute_error(test_values, pred_values)
    rmse_val = np.sqrt(mean_squared_error(test_values, pred_values))
    
    print(f"\nForecast Evaluation:")
    print(f"MAE: {mae_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")
    
    return train_data, test_data, predictions


def detect_anomalies(detector, data: pd.DataFrame, config: dict):
    """Detect anomalies in time series."""
    ts_data = TimeSeries.from_pd(data)
    detector.train(ts_data)
    predictions = detector.get_anomaly_label(ts_data)
    
    return predictions


def create_forecast_visualization(train_data, test_data, predictions, config: dict, script_dir: Path):
    """Generate forecast visualization."""
    fig, ax = plt.subplots(figsize=tuple(config.get("plotting", {}).get("figure_size", [12, 6])))
    
    train_df = train_data.to_pd()
    test_df = test_data.to_pd()
    pred_df = predictions.to_pd()
    
    ax.plot(
        train_df.index,
        train_df.values.flatten(),
        "k-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label="Train",
    )
    
    ax.plot(
        test_df.index,
        test_df.values.flatten(),
        "g-",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        alpha=config.get("plotting", {}).get("alpha", 0.8),
        label="Actual (Test)",
    )
    
    ax.plot(
        pred_df.index,
        pred_df.values.flatten(),
        "r--",
        linewidth=config.get("plotting", {}).get("linewidth", 1.5),
        label="Forecast",
    )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"Merlion {config['model']['forecaster_type']} Forecast")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / "merlion_forecast.png", dpi=300)
    print(f"Plot saved to: {output_dir / 'merlion_forecast.png'}")


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data using consolidated loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value")
    )
    
    # Convert to DataFrame format expected by Merlion
    data = pd.DataFrame({
        config["data"]["date_col"]: series.index,
        config["data"]["value_col"]: series.values
    }).set_index(config["data"]["date_col"])
    
    print(f"Loaded {len(data)} data points")
    
    # Create and fit forecaster
    if config["model"].get("forecaster_type"):
        print(f"\nCreating {config['model']['forecaster_type']} forecaster...")
        forecaster = create_forecaster(config)
        train_data, test_data, predictions = fit_and_forecast(forecaster, data, config, script_dir)
        create_forecast_visualization(train_data, test_data, predictions, config, script_dir)
    
    # Create and fit anomaly detector if configured
    if config["model"].get("detector_type"):
        print(f"\nCreating {config['model']['detector_type']} anomaly detector...")
        detector = create_anomaly_detector(config)
        anomaly_predictions = detect_anomalies(detector, data, config)
        print(f"Detected {anomaly_predictions.sum()} anomalies")
    
    print("\n Merlion analysis complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
