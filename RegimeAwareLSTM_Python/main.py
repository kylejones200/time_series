#!/usr/bin/env python3
"""Regime-Aware LSTM: LSTM with regime embeddings for time series with structural breaks."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import consolidated utilities
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    create_forecast_plot,
    save_plot,
)
from src.evaluator import Evaluator

warnings.filterwarnings("ignore")


class TimeSeriesDataset(Dataset):
    """Dataset for time series with regime information."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, regimes: np.ndarray):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences (n_samples, seq_len, n_features)
        y : np.ndarray
            Target values (n_samples,)
        regimes : np.ndarray
            Regime IDs (n_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.r = torch.tensor(regimes, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.r[idx], self.y[idx]


class VanillaLSTM(nn.Module):
    """Standard LSTM without regime information."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, *_):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class RegimeAwareLSTM(nn.Module):
    """LSTM with regime embeddings."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_regimes: int = 2,
        regime_embed_dim: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_regimes, regime_embed_dim)
        self.lstm = nn.LSTM(
            input_dim + regime_embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, regime_id):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input sequences (batch, seq_len, input_dim)
        regime_id : torch.Tensor
            Regime IDs (batch,)
        
        Returns:
        --------
        torch.Tensor
            Predictions (batch, 1)
        """
        regime_embed = self.embedding(regime_id)
        regime_expanded = regime_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        x_augmented = torch.cat([x, regime_expanded], dim=2)
        out, _ = self.lstm(x_augmented)
        return self.fc(out[:, -1, :])


def create_sequences(
    data: np.ndarray,
    regimes: np.ndarray,
    seq_len: int,
) -> tuple:
    """
    Create sequences for LSTM training.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data (n_timesteps, n_features)
    regimes : np.ndarray
        Regime IDs (n_timesteps,)
    seq_len : int
        Sequence length
    
    Returns:
    --------
    tuple
        (X, y, r) where X is sequences, y is targets, r is regimes
    """
    X, y, r = [], [], []
    
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, 0])  # Assuming first feature is target
        r.append(regimes[i + seq_len])
    
    return np.array(X), np.array(y), np.array(r)


def detect_regimes_simple(series: pd.Series, n_regimes: int = 2) -> np.ndarray:
    """
    Simple regime detection using quantiles.
    
    For production use, integrate with RegimeSwitching_Python template.
    
    Parameters:
    -----------
    series : pd.Series
        Time series
    n_regimes : int
        Number of regimes
    
    Returns:
    --------
    np.ndarray
        Regime IDs
    """
    # Simple approach: use quantiles
    quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]
    thresholds = series.quantile(quantiles).values
    
    regimes = np.zeros(len(series), dtype=int)
    for i, threshold in enumerate(thresholds):
        regimes[series.values > threshold] = i + 1
    
    return regimes


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> list:
    """
    Train LSTM model.
    
    Parameters:
    -----------
    model : nn.Module
        LSTM model
    train_loader : DataLoader
        Training data loader
    epochs : int
        Number of epochs
    learning_rate : float
        Learning rate
    device : str
        Device ('cpu' or 'cuda')
    
    Returns:
    --------
    list
        Training loss history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        for x, r, y in train_loader:
            x = x.to(device)
            r = r.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            pred = model(x, r) if isinstance(model, RegimeAwareLSTM) else model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    return loss_history


def forecast(
    model: nn.Module,
    X: np.ndarray,
    regimes: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate forecasts.
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    X : np.ndarray
        Input sequences
    regimes : np.ndarray
        Regime IDs
    device : str
        Device
    
    Returns:
    --------
    np.ndarray
        Predictions
    """
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    r_tensor = torch.tensor(regimes, dtype=torch.long).to(device)
    
    predictions = []
    with torch.no_grad():
        for i in range(len(X_tensor)):
            x = X_tensor[i : i + 1]
            r = r_tensor[i : i + 1]
            if isinstance(model, RegimeAwareLSTM):
                pred = model(x, r)
            else:
                pred = model(x)
            predictions.append(pred.cpu().numpy()[0, 0])
    
    return np.array(predictions)


def main():
    """Main execution function."""
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not installed.")
        print("Install with: pip install torch")
        sys.exit(1)
    
    script_dir = Path(__file__).parent
    config = load_config(script_dir / "config.yaml")
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    
    # Load data
    data_config = config["data"]
    repo_root = script_dir.parent
    data_path = repo_root / data_config["input_file"]
    series = load_time_series(
        str(data_path),
        date_column=data_config.get("date_column", "date"),
        value_column=data_config.get("value_column", "value"),
    )
    
    # Detect or load regimes
    model_config = config.get("model", {})
    if model_config.get("regime_detection", "simple") == "simple":
        print("Detecting regimes using quantile-based method...")
        n_regimes = model_config.get("num_regimes", 2)
        regimes = detect_regimes_simple(series, n_regimes=n_regimes)
    else:
        # Could integrate with RegimeSwitching_Python here
        raise NotImplementedError("Advanced regime detection not yet implemented")
    
    print(f"Detected {len(np.unique(regimes))} regimes")
    
    # Prepare data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    # Create sequences
    seq_len = model_config.get("sequence_length", 30)
    X, y, r = create_sequences(scaled_data, regimes, seq_len)
    
    print(f"Created {len(X)} sequences (seq_len={seq_len})")
    
    # Split data
    split_idx = int(len(X) * (1 - config["evaluation"].get("test_size", 0.2)))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    r_train, r_test = r[:split_idx], r[split_idx:]
    
    # Create data loaders
    train_dataset = TimeSeriesDataset(X_train, y_train, r_train)
    test_dataset = TimeSeriesDataset(X_test, y_test, r_test)
    
    train_loader = DataLoader(train_dataset, batch_size=model_config.get("batch_size", 32), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    input_dim = X_train.shape[2]
    hidden_dim = model_config.get("hidden_dim", 64)
    num_layers = model_config.get("num_layers", 1)
    num_regimes = len(np.unique(regimes))
    
    # Vanilla LSTM
    print("\nTraining Vanilla LSTM...")
    vanilla_model = VanillaLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    vanilla_loss = train_model(
        vanilla_model,
        train_loader,
        epochs=model_config.get("epochs", 10),
        learning_rate=model_config.get("learning_rate", 0.001),
        device=device,
    )
    
    # Regime-Aware LSTM
    print("\nTraining Regime-Aware LSTM...")
    regime_model = RegimeAwareLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_regimes=num_regimes,
        regime_embed_dim=model_config.get("regime_embed_dim", 4),
    )
    regime_loss = train_model(
        regime_model,
        train_loader,
        epochs=model_config.get("epochs", 10),
        learning_rate=model_config.get("learning_rate", 0.001),
        device=device,
    )
    
    # Evaluate
    print("\nEvaluating models...")
    vanilla_pred = forecast(vanilla_model, X_test, r_test, device=device)
    regime_pred = forecast(regime_model, X_test, r_test, device=device)
    
    # Inverse transform
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    vanilla_pred_orig = scaler.inverse_transform(vanilla_pred.reshape(-1, 1)).flatten()
    regime_pred_orig = scaler.inverse_transform(regime_pred.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    vanilla_mse = mean_squared_error(y_test_orig, vanilla_pred_orig)
    regime_mse = mean_squared_error(y_test_orig, regime_pred_orig)
    vanilla_mae = mean_absolute_error(y_test_orig, vanilla_pred_orig)
    regime_mae = mean_absolute_error(y_test_orig, regime_pred_orig)
    vanilla_r2 = r2_score(y_test_orig, vanilla_pred_orig)
    regime_r2 = r2_score(y_test_orig, regime_pred_orig)
    
    print(f"\nVanilla LSTM:")
    print(f"  MSE:  {vanilla_mse:.4f}")
    print(f"  MAE:  {vanilla_mae:.4f}")
    print(f"  R²:   {vanilla_r2:.4f}")
    
    print(f"\nRegime-Aware LSTM:")
    print(f"  MSE:  {regime_mse:.4f}")
    print(f"  MAE:  {regime_mae:.4f}")
    print(f"  R²:   {regime_r2:.4f}")
    
    improvement = ((vanilla_mse - regime_mse) / vanilla_mse) * 100
    print(f"\nImprovement: {improvement:.2f}% reduction in MSE")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training loss
    axes[0, 0].plot(vanilla_loss, label="Vanilla LSTM", alpha=0.7)
    axes[0, 0].plot(regime_loss, label="Regime-Aware LSTM", alpha=0.7)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Predictions comparison (first 100)
    n_plot = min(100, len(y_test_orig))
    axes[0, 1].plot(y_test_orig[:n_plot], label="Actual", linewidth=2, alpha=0.8)
    axes[0, 1].plot(vanilla_pred_orig[:n_plot], label="Vanilla LSTM", linestyle="--", alpha=0.8)
    axes[0, 1].plot(regime_pred_orig[:n_plot], label="Regime-Aware LSTM", linestyle=":", alpha=0.8)
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].set_title("Predictions Comparison (First 100)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot - Vanilla
    axes[1, 0].scatter(y_test_orig, vanilla_pred_orig, alpha=0.5)
    min_val = min(y_test_orig.min(), vanilla_pred_orig.min())
    max_val = max(y_test_orig.max(), vanilla_pred_orig.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
    axes[1, 0].set_xlabel("Actual")
    axes[1, 0].set_ylabel("Predicted")
    axes[1, 0].set_title("Vanilla LSTM: Actual vs Predicted")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot - Regime-Aware
    axes[1, 1].scatter(y_test_orig, regime_pred_orig, alpha=0.5)
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
    axes[1, 1].set_xlabel("Actual")
    axes[1, 1].set_ylabel("Predicted")
    axes[1, 1].set_title("Regime-Aware LSTM: Actual vs Predicted")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / config["output"].get("plot_file", "regime_lstm_comparison.png")
    save_plot(fig, plot_path, dpi=config["output"].get("dpi", 300))
    print(f"\nPlot saved to: {plot_path}")
    
    # Save results
    results_df = pd.DataFrame({
        "actual": y_test_orig,
        "vanilla_pred": vanilla_pred_orig,
        "regime_pred": regime_pred_orig,
    })
    
    csv_path = output_dir / config["output"].get("predictions_file", "regime_lstm_predictions.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Predictions saved to: {csv_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        "model": ["Vanilla LSTM", "Regime-Aware LSTM"],
        "MSE": [vanilla_mse, regime_mse],
        "MAE": [vanilla_mae, regime_mae],
        "R²": [vanilla_r2, regime_r2],
    })
    
    metrics_path = output_dir / config["output"].get("metrics_file", "regime_lstm_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

