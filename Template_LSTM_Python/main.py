#!/usr/bin/env python3
"""
LSTM for Time Series Forecasting
Long Short-Term Memory networks for time series forecasting using TensorFlow/Keras.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index, split_ts

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_sequences(data, seq_length, target_col_idx=0):
    """Create sequences for LSTM training using sliding window."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_col_idx])
    return np.array(X), np.array(y)


def build_lstm_model(seq_length, n_features, config):
    """Build LSTM model architecture."""
    model_type = config['model']['type']
    hidden_units = config['model']['hidden_units']
    dropout = config['model']['dropout']
    
    model = Sequential(name=f'{model_type}_model')
    model.add(layers.Input(shape=(seq_length, n_features)))
    
    for i, units in enumerate(hidden_units):
        return_sequences = (i < len(hidden_units) - 1)
        
        model_map = {
            'lstm': lambda u, rs: LSTM(u, return_sequences=rs),
            'gru': lambda u, rs: GRU(u, return_sequences=rs),
            'bidirectional_lstm': lambda u, rs: Bidirectional(LSTM(u, return_sequences=rs)),
        }
        
        model.add(model_map.get(model_type, model_map['lstm'])(units, return_sequences))
        
        if dropout > 0:
            model.add(Dropout(dropout))
    
    model.add(Dense(1))
    model.compile(
        optimizer=config['model']['optimizer'],
        loss=config['model']['loss'],
        metrics=['mae']
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val, config):
    """Train LSTM model with early stopping."""
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['training']['patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model and calculate metrics."""
    predictions = model.predict(X_test, verbose=0).flatten()
    
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, 0] = predictions
    predictions = scaler.inverse_transform(dummy)[:, 0]
    
    dummy[:, 0] = y_test
    y_test_inv = scaler.inverse_transform(dummy)[:, 0]
    
    mae = mean_absolute_error(y_test_inv, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
    r2 = r2_score(y_test_inv, predictions)
    
    return predictions, {'MAE': mae, 'RMSE': rmse, 'R²': r2}


def forecast_future(model, last_sequence, n_steps, scaler):
    """Generate multi-step ahead forecast."""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(n_steps):
        pred = model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)[0, 0]
        predictions.append(pred)
        new_row = current_seq[-1].copy()
        new_row[0] = pred
        current_seq = np.vstack([current_seq[1:], new_row])
    
    predictions = np.array(predictions)
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, 0] = predictions
    predictions = scaler.inverse_transform(dummy)[:, 0]
    
    return predictions


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col='date')
    
    train_df, test_df = split_ts(df, test_size=config['data']['test_size'])
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[[config['data']['value_col']]].values)
    test_scaled = scaler.transform(test_df[[config['data']['value_col']]].values)
    
    seq_length = config['model']['sequence_length']
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    
    val_size = int(len(X_train) * config['data']['val_size'])
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    model = build_lstm_model(seq_length, 1, config)
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    
    predictions, metrics = evaluate_model(model, X_test, y_test, scaler)
    
    print("\nModel Evaluation:")
    [print(f"{k}: {v:.4f}") for k, v in metrics.items()]
    
    dummy = np.zeros((len(y_test), scaler.n_features_in_))
    dummy[:, 0] = y_test
    y_test_inv = scaler.inverse_transform(dummy)[:, 0]
    
    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, config)
    
    test_dates = test_df.index[seq_length:]
    ax.plot(test_dates, y_test_inv, 'k-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Actual')
    ax.plot(test_dates, predictions, 'r--', linewidth=config['plotting']['linewidth'],
            label='Predicted')
    
    ax.set_title(config['plot_titles']['lstm_forecast'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config['plotting']['legend'])
    
    output_path = Path(__file__).parent / "outputs" / "lstm_forecast.png"
    save_plot(fig, output_path)
    plt.show()
    
    future_steps = config['model']['forecast_steps']
    last_sequence = X_test[-1]
    future_predictions = forecast_future(model, last_sequence, future_steps, scaler)
    
    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, config)
    
    ax.plot(df.index[-200:], df[config['data']['value_col']].values[-200:],
            'k-', linewidth=config['plotting']['linewidth'], alpha=config['plotting']['alpha'],
            label='Historical')
    ax.plot(test_dates, y_test_inv, 'b-', linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Test Actual')
    ax.plot(test_dates, predictions, 'b--', linewidth=config['plotting']['linewidth'],
            label='Test Predicted')
    
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                 periods=future_steps, freq='D')
    ax.plot(future_dates, future_predictions, 'r:', linewidth=config['plotting']['linewidth'],
            marker='o', markersize=config['plotting']['markersize'], label='Future Forecast')
    
    ax.set_title(config['plot_titles']['lstm_future'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config['plotting']['legend'])
    
    output_path = Path(__file__).parent / "outputs" / "lstm_future_forecast.png"
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()

