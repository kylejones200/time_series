import numpy as np
import pandas as pd
import pmdarima as pmd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


def prepare_data(data: np.ndarray, n_steps: int):
    """Prepare time series data for supervised learning."""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


def load_and_preprocess_data(filepath: str, n_steps: int = 30, test_size: float = 0.2):
    """
    Load and preprocess time series data.

    IMPORTANT: For predictive modeling, data is split BEFORE scaling to prevent
    data leakage. The scaler is fit only on training data.

    Parameters:
    -----------
    filepath : str
        Path to CSV file with 'date' and 'values' columns
    n_steps : int
        Number of time steps for sequence creation
    test_size : float
        Proportion of data to use for testing (default: 0.2)

    Returns:
    --------
    X_train, y_train : arrays
        Training sequences and targets
    X_test, y_test : arrays
        Testing sequences and targets
    scaler : fitted MinMaxScaler
        Scaler fitted on training data only
    """
    df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)

    # IMPORTANT: Split data BEFORE scaling to avoid data leakage
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Fit scaler on training data only
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[["values"]])
    test_scaled = scaler.transform(test_df[["values"]])  # Use training statistics

    # Prepare sequences
    X_train, y_train = prepare_data(train_scaled, n_steps)
    X_test, y_test = prepare_data(test_scaled, n_steps)

    return X_train, y_train, X_test, y_test, scaler


def build_lstm_model(n_steps: int):
    """Build a simple LSTM model."""
    model = Sequential(
        [
            LSTM(
                50, activation="relu", return_sequences=True, input_shape=(n_steps, 1)
            ),
            LSTM(50, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


class PyTorchLSTM(nn.Module):
    """PyTorch implementation of an LSTM for time series forecasting."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(PyTorchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def train_pytorch_lstm(model, train_loader, criterion, optimizer, epochs=10):
    """Train a PyTorch LSTM model."""
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def auto_arima_forecast(data: pd.Series, seasonal: bool = False, m: int = 1):
    """Fit an automatic ARIMA model using pmdarima."""
    model = pmd.auto_arima(
        data,
        seasonal=seasonal,
        m=m,
        trace=True,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
    )
    return model


def arima_forecast(data: pd.Series, order: tuple, seasonal_order: tuple = (0, 0, 0, 0)):
    """Fit an ARIMA model with specified order using statsmodels."""
    model = ARIMA(data, order=order, seasonal_order=seasonal_order)
    result = model.fit()
    return result
