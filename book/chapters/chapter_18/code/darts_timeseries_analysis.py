"""
Darts Time Series Forecasting Examples

This module demonstrates various forecasting models using the Darts library
for time series analysis. It includes examples of traditional models (ARIMA,
Exponential Smoothing) and modern deep learning models (NBEATS, FFT).

IMPORTANT: Before running, set your FRED API key as an environment variable:
    export FRED_API_KEY="your_api_key_here"
    
Or modify the FRED_API_KEY variable below.

Data Leakage Prevention:
- Data is split BEFORE scaling
- Scalers are fit only on training data
- Validation data is transformed using training statistics
"""

import warnings
import os
import logging
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.models import (
    ARIMA,
    ExponentialSmoothing,
    NBEATSModel,
    FFT,
    LightGBMModel,
    RNNModel
)
from darts.metrics import mae, mape, r2_score
from darts.utils.callbacks import TFMProgressBar

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# Get API key from environment variable or set to empty string
FRED_API_KEY = os.getenv("FRED_API_KEY", "")


def fetch_fred_data(series_id, api_key=None, start_date='2000-01-01'):
    """
    Fetch data from FRED API and return as a Darts TimeSeries.
    
    Parameters:
    -----------
    series_id : str
        FRED series ID (e.g., 'T10Y2Y' for yield spread)
    api_key : str, optional
        FRED API key. If None, uses FRED_API_KEY global variable.
    start_date : str
        Start date for data retrieval (default: '2000-01-01')
    
    Returns:
    --------
    TimeSeries
        Darts TimeSeries object with the data
    """
    if api_key is None:
        api_key = FRED_API_KEY
    
    if not api_key:
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY environment variable "
            "or pass api_key parameter. Get a free key at: "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': datetime.now().strftime('%Y-%m-%d'),
    }
    url = "https://api.stlouisfed.org/fred/series/observations"
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"FRED API request failed with status code {response.status_code}")
    
    data = response.json()
    observations = data['observations']
    df = pd.DataFrame(observations)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Handle missing values by forward filling
    df['value'] = df['value'].ffill()
    df = df.dropna()
    df = df.sort_values('date')
    
    return TimeSeries.from_dataframe(df, 'date', 'value')


def prepare_data_without_leakage(series, split_date='2020-01-01'):
    """
    Prepare data for modeling with proper train/test split to avoid data leakage.
    
    IMPORTANT: Data is split BEFORE scaling. The scaler is fit only on training data.
    
    Parameters:
    -----------
    series : TimeSeries
        Raw time series data
    split_date : str or pd.Timestamp
        Date to split train and validation sets
    
    Returns:
    --------
    train_scaled : TimeSeries
        Scaled training data
    val_scaled : TimeSeries
        Scaled validation data (using training statistics)
    scaler : Scaler
        Fitted scaler (fitted on training data only)
    """
    # Split data BEFORE scaling to avoid data leakage
    train_raw, val_raw = series.split_before(pd.Timestamp(split_date))
    
    # Handle missing values and scale the data
    filler = MissingValuesFiller()
    scaler = Scaler()
    
    # Fit filler and scaler on training data only
    train_filled = filler.fit_transform(train_raw)
    train_scaled = scaler.fit_transform(train_filled)
    
    # Transform validation data using training statistics
    val_filled = filler.transform(val_raw)
    val_scaled = scaler.transform(val_filled)
    
    return train_scaled, val_scaled, scaler


def plot_forecast_comparison(train, val, pred, title, filename=None):
    """
    Plot training data, validation data, and predictions.
    
    Parameters:
    -----------
    train : TimeSeries
        Training data
    val : TimeSeries
        Validation data
    pred : TimeSeries
        Predictions
    title : str
        Plot title
    filename : str, optional
        Filename to save plot (if None, plot is shown but not saved)
    """
    plt.figure(figsize=(12, 6))
    train.plot(label="Train", color='blue', alpha=0.7)
    val.plot(label="Validation", color='green', alpha=0.7)
    pred.plot(label="Prediction", color='red', linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def example_exponential_smoothing(series):
    """Example: Exponential Smoothing forecast."""
    print("\n" + "="*70)
    print("Example 1: Exponential Smoothing")
    print("="*70)
    
    train, val = series.split_before(0.8)
    
    model = ExponentialSmoothing()
    model.fit(train)
    forecast = model.predict(len(val))
    
    mape_score = mape(val, forecast)
    mae_score = mae(val, forecast)
    
    print(f"MAPE: {mape_score:.2f}%")
    print(f"MAE: {mae_score:.4f}")
    
    plot_forecast_comparison(train, val, forecast, 
                            "Exponential Smoothing Forecast",
                            "exponential_smoothing_forecast.png")
    
    return model, forecast


def example_arima(series):
    """Example: ARIMA forecast."""
    print("\n" + "="*70)
    print("Example 2: ARIMA")
    print("="*70)
    
    train, val = series.split_before(0.8)
    
    model = ARIMA(p=1, d=1, q=1)
    model.fit(train)
    forecast = model.predict(len(val))
    
    mape_score = mape(val, forecast)
    mae_score = mae(val, forecast)
    
    print(f"MAPE: {mape_score:.2f}%")
    print(f"MAE: {mae_score:.4f}")
    
    plot_forecast_comparison(train, val, forecast,
                            "ARIMA Forecast",
                            "arima_forecast.png")
    
    return model, forecast


def example_nbeats(series, api_key=None):
    """Example: NBEATS deep learning model with proper data leakage prevention."""
    print("\n" + "="*70)
    print("Example 3: NBEATS (Neural Basis Expansion Analysis)")
    print("="*70)
    
    # Prepare data with proper split to avoid leakage
    train_scaled, val_scaled, scaler = prepare_data_without_leakage(
        series, split_date='2020-01-01'
    )
    
    # Configure PyTorch settings
    torch_kwargs = {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }
    
    # Create and train NBEATS model
    model = NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=7,
        n_epochs=50,  # Reduced for faster execution
        random_state=42,
        **torch_kwargs
    )
    
    print("Training NBEATS model...")
    model.fit(train_scaled, val_series=val_scaled)
    
    # Make predictions
    forecast_scaled = model.predict(len(val_scaled))
    
    # Inverse transform to original scale
    train_original = scaler.inverse_transform(train_scaled)
    val_original = scaler.inverse_transform(val_scaled)
    forecast_original = scaler.inverse_transform(forecast_scaled)
    
    mape_score = mape(val_original, forecast_original)
    mae_score = mae(val_original, forecast_original)
    r2 = r2_score(val_original, forecast_original)
    
    print(f"MAPE: {mape_score:.2f}%")
    print(f"MAE: {mae_score:.4f}")
    print(f"R²: {r2:.4f}")
    
    plot_forecast_comparison(train_original, val_original, forecast_original,
                            "NBEATS Forecast",
                            "nbeats_forecast.png")
    
    return model, forecast_original


def example_fft(series):
    """Example: FFT (Fast Fourier Transform) forecast."""
    print("\n" + "="*70)
    print("Example 4: FFT (Fast Fourier Transform)")
    print("="*70)
    
    train, val = series.split_before(0.8)
    
    model = FFT(nr_freqs_to_keep=20)
    model.fit(train)
    forecast = model.predict(len(val))
    
    mape_score = mape(val, forecast)
    mae_score = mae(val, forecast)
    
    print(f"MAPE: {mape_score:.2f}%")
    print(f"MAE: {mae_score:.4f}")
    
    plot_forecast_comparison(train, val, forecast,
                            "FFT Forecast",
                            "fft_forecast.png")
    
    return model, forecast


def example_lightgbm(series):
    """Example: LightGBM machine learning model."""
    print("\n" + "="*70)
    print("Example 5: LightGBM")
    print("="*70)
    
    train, val = series.split_before(0.8)
    
    model = LightGBMModel(lags=30)
    model.fit(train)
    forecast = model.predict(len(val))
    
    mape_score = mape(val, forecast)
    mae_score = mae(val, forecast)
    
    print(f"MAPE: {mape_score:.2f}%")
    print(f"MAE: {mae_score:.4f}")
    
    plot_forecast_comparison(train, val, forecast,
                            "LightGBM Forecast",
                            "lightgbm_forecast.png")
    
    return model, forecast


def example_lstm(series):
    """Example: LSTM (Long Short-Term Memory) neural network."""
    print("\n" + "="*70)
    print("Example 6: LSTM (Long Short-Term Memory)")
    print("="*70)
    
    train, val = series.split_before(0.8)
    
    model = RNNModel(
        model="LSTM",
        input_chunk_length=30,
        output_chunk_length=7,
        n_epochs=50,
        random_state=42
    )
    
    print("Training LSTM model...")
    model.fit(train, val_series=val)
    forecast = model.predict(len(val))
    
    mape_score = mape(val, forecast)
    mae_score = mae(val, forecast)
    
    print(f"MAPE: {mape_score:.2f}%")
    print(f"MAE: {mae_score:.4f}")
    
    plot_forecast_comparison(train, val, forecast,
                            "LSTM Forecast",
                            "lstm_forecast.png")
    
    return model, forecast


def main():
    """
    Main function demonstrating various Darts forecasting models.
    
    This example uses the 10-Year Treasury Constant Maturity Minus 2-Year
    Treasury Constant Maturity (T10Y2Y) series from FRED, which measures
    the yield spread - an important financial indicator.
    """
    print("="*70)
    print("Darts Time Series Forecasting Examples")
    print("="*70)
    print("\nThis script demonstrates various forecasting models using Darts.")
    print("Using T10Y2Y (Treasury Yield Spread) data from FRED.\n")
    
    # Check for API key
    if not FRED_API_KEY:
        print("WARNING: FRED_API_KEY not set. Using synthetic data instead.")
        print("To use real FRED data, set the environment variable:")
        print("  export FRED_API_KEY='your_key_here'\n")
        
        # Generate synthetic data for demonstration
        dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='D')
        values = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        series = TimeSeries.from_times_and_values(dates, values)
    else:
        try:
            series = fetch_fred_data('T10Y2Y', api_key=FRED_API_KEY)
            print(f"Successfully loaded {len(series)} data points from FRED.")
        except Exception as e:
            print(f"Error fetching FRED data: {e}")
            print("Using synthetic data instead.\n")
            dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='D')
            values = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            series = TimeSeries.from_times_and_values(dates, values)
    
    # Run examples
    try:
        example_exponential_smoothing(series)
    except Exception as e:
        print(f"Error in Exponential Smoothing example: {e}")
    
    try:
        example_arima(series)
    except Exception as e:
        print(f"Error in ARIMA example: {e}")
    
    try:
        example_fft(series)
    except Exception as e:
        print(f"Error in FFT example: {e}")
    
    try:
        example_lightgbm(series)
    except Exception as e:
        print(f"Error in LightGBM example: {e}")
    
    # Deep learning models (may take longer)
    try:
        example_nbeats(series)
    except Exception as e:
        print(f"Error in NBEATS example: {e}")
    
    try:
        example_lstm(series)
    except Exception as e:
        print(f"Error in LSTM example: {e}")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
