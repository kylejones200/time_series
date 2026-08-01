# Forecasting the U.S. Treasury Yield Spread using Darts

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests

from darts import TimeSeries
from darts.models import (
    ARIMA,
    ExponentialSmoothing,
    LightGBMModel,
    RNNModel,
    FFT,
    NBEATSModel,
)
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.metrics import mae, mape, r2_score
from darts.utils.callbacks import TFMProgressBar


# Fetch and clean data from FRED
def fetch_fred_series(series_id, api_key, start="2000-01-01"):
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": datetime.now().strftime("%Y-%m-%d"),
    }
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception(f"FRED API error {r.status_code}")
    df = pd.DataFrame(r.json()["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce").ffill()
    return TimeSeries.from_dataframe(df.sort_values("date"), "date", "value")


# Plot forecast vs actual
def plot_forecast(series, forecast, title, filename):
    plt.figure(figsize=(12, 6))
    series.plot(label="Actual")
    forecast.plot(label="Forecast")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# Torch kwargs for NBEATS
def torch_config():
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }


if __name__ == "__main__":
    api_key = "8f058d10ec8c788296c040ea09e634d5"
    series_id = "T10Y2Y"

    # Get data
    series = fetch_fred_series(series_id, api_key)
    series = MissingValuesFiller().transform(series)
    scaler = Scaler()
    series = scaler.fit_transform(series)

    # Split
    train, val = series.split_before(pd.Timestamp("2020-01-01"))

    # ARIMA
    model = ARIMA(1, 1, 1)
    model.fit(train)
    forecast = model.predict(len(val))
    forecast = scaler.inverse_transform(forecast)
    actual = scaler.inverse_transform(series)
    plot_forecast(actual, forecast, "ARIMA Forecast", "ARIMA.png")

    # Exponential Smoothing
    model = ExponentialSmoothing()
    model.fit(train)
    forecast = model.predict(len(val))
    forecast = scaler.inverse_transform(forecast)
    plot_forecast(actual, forecast, "Exponential Smoothing", "ExponentialSmoothing.png")

    # LightGBM
    model = LightGBMModel(lags=30)
    model.fit(train)
    forecast = model.predict(len(val))
    forecast = scaler.inverse_transform(forecast)
    plot_forecast(actual, forecast, "LightGBM Forecast", "LightGBM.png")

    # LSTM
    model = RNNModel(
        model="LSTM", input_chunk_length=30, output_chunk_length=7, n_epochs=50
    )
    model.fit(train)
    forecast = model.predict(len(val))
    forecast = scaler.inverse_transform(forecast)
    plot_forecast(actual, forecast, "LSTM Forecast", "LSTM.png")

    # NBEATS
    model = NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=7,
        n_epochs=50,
        random_state=42,
        **torch_config(),
    )
    model.fit(train, val_series=val)
    forecast = model.predict(len(val))
    forecast = scaler.inverse_transform(forecast)
    plot_forecast(actual, forecast, "NBEATS Forecast", "NBEATS.png")

    # FFT
    model = FFT()
    model.fit(train)
    forecast = model.predict(len(val))
    forecast = scaler.inverse_transform(forecast)
    plot_forecast(actual, forecast, "FFT Forecast", "FFT.png")
