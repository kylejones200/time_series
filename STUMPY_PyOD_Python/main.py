#!/usr/bin/env python3
"""Anomaly detection template using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from statsmodels.tsa.seasonal import STL

try:
    import stumpy
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
except ImportError:  # pragma: no cover - optional dependencies
    stumpy = None  # type: ignore
    IForest = None  # type: ignore
    LOF = None  # type: ignore
    OCSVM = None  # type: ignore


@dataclass
class STLConfig:
    """STL configuration."""
    enabled: bool
    season: int
    z_threshold: float
    output_plot: Path


@dataclass
class AutoencoderConfig:
    """Autoencoder configuration."""
    enabled: bool
    window: int
    batch_size: int
    epochs: int
    learning_rate: float
    z_threshold: float
    output_plot: Path
    error_plot: Path


@dataclass
class StumpyConfig:
    """STUMPY configuration."""
    enabled: bool
    window: int
    percentile: float


@dataclass
class PyODConfig:
    """PyOD configuration."""
    enabled: bool
    method: str
    contamination: float


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    stl: STLConfig
    autoencoder: AutoencoderConfig
    stumpy: StumpyConfig
    pyod: PyODConfig
    output_dir: Path
    colors: dict


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    repo_root = script_dir.parent
    output_dir = ensure_output_dir(Path(script_dir) / config_dict["output"]["output_dir"])
    
    stl_cfg = config_dict["methods"]["stl"]
    ae_cfg = config_dict["methods"]["autoencoder"]
    stumpy_cfg = config_dict["methods"]["stumpy"]
    pyod_cfg = config_dict["methods"]["pyod"]
    
    colors = {
        "series": config_dict["plotting"].get("history_color", "#2f2f2f"),
        "stl": config_dict["plotting"].get("stl_color", "#444444"),
        "anomaly": config_dict["plotting"].get("anomaly_color", "#c70039"),
    }
    
    return Config(
        data_path=repo_root / "data" / config_dict["data"]["input_file"],
        date_col=config_dict["data"]["date_col"],
        value_col=config_dict["data"]["value_col"],
        freq=config_dict["data"].get("freq", "MS"),
        stl=STLConfig(
            enabled=bool(stl_cfg.get("enabled", True)),
            season=int(stl_cfg.get("season", 12)),
            z_threshold=float(stl_cfg.get("z_threshold", 3.0)),
            output_plot=output_dir / stl_cfg.get("output_plot", "eia_anomaly_stl.png"),
        ),
        autoencoder=AutoencoderConfig(
            enabled=bool(ae_cfg.get("enabled", True)),
            window=int(ae_cfg.get("window", 24)),
            batch_size=int(ae_cfg.get("batch_size", 32)),
            epochs=int(ae_cfg.get("epochs", 200)),
            learning_rate=float(ae_cfg.get("learning_rate", 1e-3)),
            z_threshold=float(ae_cfg.get("z_threshold", 3.0)),
            output_plot=output_dir / ae_cfg.get("output_plot", "eia_anomaly_autoencoder.png"),
            error_plot=output_dir / ae_cfg.get("error_plot", "eia_anomaly_autoencoder_error.png"),
        ),
        stumpy=StumpyConfig(
            enabled=bool(stumpy_cfg.get("enabled", False)),
            window=int(stumpy_cfg.get("window", 50)),
            percentile=float(stumpy_cfg.get("percentile", 98)),
        ),
        pyod=PyODConfig(
            enabled=bool(pyod_cfg.get("enabled", False)),
            method=pyod_cfg.get("method", "IForest"),
            contamination=float(pyod_cfg.get("contamination", 0.1)),
        ),
        output_dir=output_dir,
        colors=colors,
    )


def load_series(config: Config) -> pd.Series:
    """Load time series using consolidated loader."""
    from src import load_time_series
    series = load_time_series(
        str(config.data_path),
        date_column=config.date_col,
        value_column=config.value_col
    )
    
    if config.freq:
        series = series.asfreq(config.freq)
    
    return series.astype(float)


def run_stl(series: pd.Series, config: Config) -> pd.Series:
    """Run STL decomposition anomaly detection."""
    stl = STL(series, period=config.stl.season, robust=True).fit()
    resid = pd.Series(stl.resid, index=series.index)
    mu = resid.mean()
    sigma = resid.std(ddof=1) or 1.0
    z_scores = (resid - mu) / sigma
    anomalies = z_scores.abs() > config.stl.z_threshold
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        series.index,
        series.values,
        color=config.colors["stl"],
        alpha=0.8,
        label="Series",
    )
    ax.scatter(
        series.index[anomalies],
        series[anomalies],
        color=config.colors["anomaly"],
        s=24,
        label="STL anomaly",
    )
    ax.set_title("STL residual z-score anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, config.stl.output_plot, dpi=300)
    plt.close(fig)
    
    print(f" STL anomalies saved -> {config.stl.output_plot}")
    print(f"  Anomalies detected: {int(anomalies.sum())}")
    return resid


class ResidualAutoencoder(nn.Module):
    """Residual autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """Build sliding windows from array."""
    if len(arr) < window:
        return np.empty((0, window), dtype=float)
    return np.stack([arr[i : i + window] for i in range(len(arr) - window + 1)], axis=0)


def train_autoencoder(
    residuals: pd.Series, config: Config
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Train autoencoder and detect anomalies."""
    residuals = residuals.dropna()
    mu = residuals.mean()
    sigma = residuals.std(ddof=1) or 1.0
    zres = (residuals - mu) / sigma
    
    windows = build_windows(zres.values.astype(np.float32), config.autoencoder.window)
    if windows.size == 0:
        raise ValueError("Time series too short for configured autoencoder window.")
    
    n = len(windows)
    lo, hi = int(0.1 * n), int(0.9 * n)
    train_windows = windows[lo:hi]
    
    device = torch.device("cpu")
    model = ResidualAutoencoder(input_dim=train_windows.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.autoencoder.learning_rate
    )
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_windows))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.autoencoder.batch_size, shuffle=True
    )
    
    model.train()
    for _ in range(config.autoencoder.epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch.float())
            loss = criterion(recon, batch.float())
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        all_windows = torch.from_numpy(windows).float().to(device)
        recon = model(all_windows).cpu().numpy()
    errors = np.mean((recon - windows) ** 2, axis=1)
    error_index = residuals.index[config.autoencoder.window - 1 :]
    error_series = pd.Series(errors, index=error_index)
    
    err_mu = error_series.mean()
    err_sigma = error_series.std(ddof=1) or 1.0
    z_scores = (error_series - err_mu) / err_sigma
    anomalies = z_scores > config.autoencoder.z_threshold
    
    # Plot residual anomalies
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        residuals.index,
        residuals.values,
        label="STL residual",
        color=config.colors["series"],
        alpha=0.8,
    )
    ax.scatter(
        error_series.index[anomalies],
        residuals.reindex(error_series.index)[anomalies],
        color=config.colors["anomaly"],
        s=24,
        label="AE anomaly",
    )
    ax.set_title("Autoencoder residual anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, config.autoencoder.output_plot, dpi=300)
    plt.close(fig)
    
    # Plot reconstruction error
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(
        error_series.index,
        error_series.values,
        color="tab:blue",
        label="Reconstruction error",
        alpha=0.8,
    )
    ax.axhline(
        err_mu + config.autoencoder.z_threshold * err_sigma,
        color=config.colors["anomaly"],
        linestyle="--",
        label="Threshold",
        lw=2,
    )
    ax.set_title("Autoencoder reconstruction error")
    ax.set_xlabel("Date")
    ax.set_ylabel("Error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, config.autoencoder.error_plot, dpi=300)
    plt.close(fig)
    
    print(f" Autoencoder anomalies saved -> {config.autoencoder.output_plot}")
    print(f" Error diagnostics saved -> {config.autoencoder.error_plot}")
    print(f"  AE anomalies detected: {int(anomalies.sum())}")
    
    return error_series, z_scores, anomalies


def run_stumpy(series: pd.Series, config: Config) -> None:
    """Run STUMPY matrix profile anomaly detection."""
    if not config.stumpy.enabled:
        return
    if stumpy is None:
        print("Warning: stumpy not available. Skipping STUMPY anomaly detection.")
        return
    
    matrix_profile = stumpy.stump(series.values, m=config.stumpy.window)[:, 0]
    threshold = np.percentile(matrix_profile, config.stumpy.percentile)
    anomalies = matrix_profile > threshold
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(
        series.index, series.values, label="Series", color=config.colors["series"], alpha=0.8
    )
    axes[0].scatter(
        series.index[anomalies],
        series.values[anomalies],
        color=config.colors["anomaly"],
        s=20,
        label="Matrix profile anomaly",
    )
    axes[0].legend(loc="best")
    axes[0].set_title("STUMPY matrix profile anomalies")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(
        series.index[: len(matrix_profile)], matrix_profile, label="Matrix profile", color="tab:blue", alpha=0.8
    )
    axes[1].axhline(
        threshold, color=config.colors["anomaly"], linestyle="--", label="Threshold", lw=2
    )
    axes[1].legend(loc="best")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Matrix Profile")
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    path = config.output_dir / "stumpy_matrix_profile.png"
    save_plot(fig, path, dpi=300)
    plt.close(fig)
    print(f" STUMPY matrix profile saved -> {path}")


def run_pyod(series: pd.Series, config: Config) -> None:
    """Run PyOD anomaly detection."""
    if not config.pyod.enabled:
        return
    if IForest is None or LOF is None or OCSVM is None:
        print("Warning: PyOD not available. Skipping PyOD anomaly detection.")
        return
    
    method_map = {
        "IForest": IForest(contamination=config.pyod.contamination, random_state=42),
        "LOF": LOF(contamination=config.pyod.contamination),
        "OCSVM": OCSVM(contamination=config.pyod.contamination),
    }
    model = method_map.get(config.pyod.method, list(method_map.values())[0])
    model.fit(series.values.reshape(-1, 1))
    preds = model.predict(series.values.reshape(-1, 1)) == 1
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, label="Series", color=config.colors["series"], alpha=0.8)
    ax.scatter(
        series.index[preds],
        series.values[preds],
        label=f"{config.pyod.method} anomaly",
        color=config.colors["anomaly"],
        s=24,
    )
    ax.legend(loc="best")
    ax.set_title(f"PyOD ({config.pyod.method}) anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    path = config.output_dir / f"pyod_{config.pyod.method.lower()}_anomalies.png"
    save_plot(fig, path, dpi=300)
    plt.close(fig)
    print(f" PyOD anomalies saved -> {path}")


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Load series
    series = load_series(config)
    
    print(
        f"Loaded series with {len(series)} points from {series.index.min().date()} to {series.index.max().date()}"
    )
    
    residuals = None
    if config.stl.enabled:
        print("\nRunning STL anomaly detection...")
        residuals = run_stl(series, config)
    
    if config.autoencoder.enabled:
        print("\nTraining autoencoder for anomaly detection...")
        residuals_for_ae = (
            residuals if residuals is not None else series - series.mean()
        )
        train_autoencoder(residuals_for_ae, config)
    
    if config.stumpy.enabled:
        print("\nRunning STUMPY matrix profile anomaly detection...")
        run_stumpy(series, config)
    
    if config.pyod.enabled:
        print("\nRunning PyOD anomaly detection...")
        run_pyod(series, config)
    
    print("\n Anomaly detection pipeline complete")


if __name__ == "__main__":
    main()
