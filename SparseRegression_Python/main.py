#!/usr/bin/env python3
"""Feature importance and supervised-learning visuals using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    season: int
    n_splits: int
    window_size: int
    output_dir: Path
    rf_plot: Path
    supervised_forecast_plot: Path
    supervised_importance_plot: Path
    seasonal_pattern_plot: Path
    fuel_mix_plot: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    repo_root = script_dir.parent
    data_path = repo_root / "data" / config_dict["data"]["input_file"]
    output_dir = ensure_output_dir(Path(script_dir) / config_dict["output"]["output_dir"])
    
    return Config(
        data_path=data_path,
        date_col=config_dict["data"]["date_col"],
        value_col=config_dict["data"]["value_col"],
        season=int(config_dict["feature_importance"]["season"]),
        n_splits=int(config_dict["feature_importance"]["n_splits"]),
        window_size=int(config_dict["sliding_window"]["window_size"]),
        output_dir=output_dir,
        rf_plot=output_dir / config_dict["output"]["random_forest_plot"],
        supervised_forecast_plot=output_dir / config_dict["output"]["supervised_forecast"],
        supervised_importance_plot=output_dir / config_dict["output"]["supervised_importance"],
        seasonal_pattern_plot=output_dir / config_dict["output"]["seasonal_pattern"],
        fuel_mix_plot=output_dir / config_dict["output"]["fuel_mix_plot"],
    )


def load_dataframe(config: Config) -> pd.DataFrame:
    """Load DataFrame from CSV."""
    from src import load_time_series
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")
    
    df = pd.read_csv(config.data_path, skiprows=4)
    df = df.rename(columns=lambda c: c.strip().lower())
    if config.date_col.lower() not in df.columns:
        raise ValueError("Date column not present in CSV")
    df[config.date_col.lower()] = pd.to_datetime(df[config.date_col.lower()], errors="coerce")
    df = df.dropna(subset=[config.date_col.lower()])
    return df.sort_values(config.date_col.lower()).reset_index(drop=True)


def build_features(series: pd.Series, season: int) -> pd.DataFrame:
    """Build feature matrix."""
    df = pd.DataFrame({"y": series})
    for k in range(1, season + 1):
        df[f"lag{k}"] = df["y"].shift(k)
    for window in (3, 6, 12):
        df[f"roll_mean_{window}"] = df["y"].rolling(window).mean()
        df[f"roll_std_{window}"] = df["y"].rolling(window).std()
    m = (
        df.index.month
        if isinstance(df.index, pd.DatetimeIndex)
        else pd.Series(df.index, index=df.index) % 12 + 1
    )
    df["sin12"] = np.sin(2 * np.pi * m / 12.0)
    df["cos12"] = np.cos(2 * np.pi * m / 12.0)
    return df.dropna()


def feature_importance_pipeline(df: pd.DataFrame, config: Config) -> None:
    """Run feature importance pipeline."""
    df = df.set_index(config.date_col.lower())
    target = df[config.value_col.lower()].astype(float)
    features_df = build_features(target, config.season)
    
    features = features_df.columns.drop("y")
    X = features_df[features]
    y = features_df["y"]
    
    # Time-aware split
    idx = np.arange(len(X))
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    maes = []
    
    for train_idx, test_idx in splitter.split(idx):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae_val = mean_absolute_error(y_test, y_pred)
        maes.append(mae_val)
    
    mean_mae = float(np.mean(maes)) if maes else float("nan")
    print(f"Random Forest - Mean MAE: {mean_mae:.3f}")
    
    # Feature importance
    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(X, y)
    feature_importance = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(20)
    ax.barh(range(len(top_features)), top_features.values, alpha=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 20 Feature Importance (Random Forest)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, config.supervised_importance_plot, dpi=300)
    plt.close(fig)
    print(f" Feature importance plot saved -> {config.supervised_importance_plot}")


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Load DataFrame
    df = load_dataframe(config)
    print(f"Loaded {len(df)} data points")
    
    # Feature importance pipeline
    print("\nRunning feature importance pipeline...")
    feature_importance_pipeline(df, config)
    
    print("\n Sparse regression analysis complete")


if __name__ == "__main__":
    main()
