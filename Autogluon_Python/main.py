#!/usr/bin/env python3
"""
AutoGluon for Time Series Forecasting
Loads CSV, fits TimeSeriesPredictor, evaluates on hold-out window.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataclasses import dataclass
from typing import Optional

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
from src.config import parse_common_config

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


@dataclass
class Config:
    """Configuration dataclass for this template."""
    data_path: Path
    date_col: str
    value_col: str
    item_id_col: Optional[str]
    default_item_id: str
    frequency: Optional[str]
    prediction_length: int
    holdout_length: int
    model_path: Path
    eval_metric: str
    presets: Optional[str]
    hyperparameters: dict
    num_val_windows: int
    save_leaderboard: bool
    output_dir: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    common = parse_common_config(config_dict, script_dir)

    if not common.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {common.data_path}")

    return Config(
        data_path=common.data_path,
        date_col=common.date_col,
        value_col=common.value_col,
        item_id_col=config_dict["data"].get("item_id_col"),
        default_item_id=config_dict["data"].get("default_item_id", "series_1"),
        frequency=config_dict["data"].get("frequency"),
        prediction_length=config_dict["model"]["prediction_length"],
        holdout_length=config_dict["model"].get(
            "holdout_length", config_dict["model"]["prediction_length"]
        ),
        model_path=Path(script_dir) / config_dict["model"].get("model_path", "autogluon_model"),
        eval_metric=config_dict["model"].get("eval_metric", "MAPE"),
        presets=config_dict["model"].get("presets"),
        hyperparameters=config_dict["model"].get("hyperparameters", {}),
        num_val_windows=config_dict["model"].get("num_val_windows", 1),
        save_leaderboard=config_dict["model"].get("save_leaderboard", True),
        output_dir=common.output_dir,
    )


def load_timeseries_dataframe(config: Config) -> TimeSeriesDataFrame:
    """Load data into AutoGluon TimeSeriesDataFrame format."""
    df = pd.read_csv(config.data_path, encoding="utf-8")
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified date/value columns not found in CSV")
    
    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.value_col])
    
    if config.item_id_col:
        df = df.rename(columns={
            config.date_col: "timestamp",
            config.value_col: "target",
            config.item_id_col: "item_id",
        })
    else:
        df = df.rename(columns={
            config.date_col: "timestamp",
            config.value_col: "target",
        })
        df["item_id"] = config.default_item_id
    
    df = df[["item_id", "timestamp", "target"]].sort_values(["item_id", "timestamp"])
    
    return TimeSeriesDataFrame(df)


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Load data
    print("Loading data into TimeSeriesDataFrame...")
    ts_df = load_timeseries_dataframe(config)
    print(f"Loaded {len(ts_df)} data points")
    
    # Split train/test
    train = ts_df.iloc[: -config.holdout_length]
    test = ts_df.iloc[-config.holdout_length :]
    
    print(f"Train: {len(train)} points, Test: {len(test)} points")
    
    # Create predictor
    print("\nFitting AutoGluon TimeSeriesPredictor...")
    predictor = TimeSeriesPredictor(
        path=str(config.model_path),
        target="target",
        eval_metric=config.eval_metric,
        prediction_length=config.prediction_length,
        freq=config.frequency,
    )
    
    # Get additional fit parameters from config
    fit_params = {
        "presets": config.presets,
        "hyperparameters": config.hyperparameters,
        "num_val_windows": config.num_val_windows,
    }
    
    # Add optional parameters
    if config_dict["model"].get("enable_ensemble") is not None:
        fit_params["enable_ensemble"] = config_dict["model"]["enable_ensemble"]
    if config_dict["model"].get("refit_every_n_windows") is not None:
        fit_params["refit_every_n_windows"] = config_dict["model"]["refit_every_n_windows"]
    if config_dict["model"].get("refit_full") is not None:
        fit_params["refit_full"] = config_dict["model"]["refit_full"]
    if config_dict["model"].get("random_seed") is not None:
        fit_params["random_seed"] = config_dict["model"]["random_seed"]
    
    predictor.fit(train, **fit_params)
    
    # Generate forecast with quantiles if specified
    print("Generating forecast...")
    quantile_levels = config_dict["model"].get("quantile_levels")
    if quantile_levels:
        forecast = predictor.predict(train, quantile_levels=quantile_levels)
        print(f"Generated quantile forecast with levels: {quantile_levels}")
    else:
        forecast = predictor.predict(train)
    
    # Evaluate
    if config.save_leaderboard:
        leaderboard = predictor.leaderboard(train, silent=True)
        print(f"\nModel Leaderboard:")
        print(leaderboard)
    
    # Create visualization
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique item IDs
    item_ids = forecast.index.get_level_values("item_id").unique()
    
    # Plot each series
    for item_id in item_ids:
        forecast_series = forecast.loc[item_id, "mean"]
        train_series = train.loc[item_id, "target"]
        test_series = test.loc[item_id, "target"] if len(test) > 0 else None
        
        # Plot training data
        ax.plot(
            train_series.index,
            train_series.values,
            "k-",
            lw=1.5,
            label=f"{item_id} (Train)" if len(item_ids) > 1 else "Train",
            alpha=0.7,
        )
        
        # Plot test data if available
        if test_series is not None and len(test_series) > 0:
            ax.plot(
                test_series.index,
                test_series.values,
                "g-",
                lw=1.5,
                label=f"{item_id} (Test)" if len(item_ids) > 1 else "Test",
                alpha=0.7,
            )
        
        # Plot forecast
        ax.plot(
            forecast_series.index,
            forecast_series.values,
            "r--",
            lw=2.0,
            label=f"{item_id} Forecast" if len(item_ids) > 1 else "AutoGluon Forecast",
            alpha=0.8,
        )
        
        # Plot quantile intervals if available
        quantile_levels = config_dict["model"].get("quantile_levels")
        if quantile_levels and len(quantile_levels) > 0:
            # Use 0.1 and 0.9 for 80% interval, or closest available
            lower_q = min(quantile_levels, key=lambda x: abs(x - 0.1))
            upper_q = min(quantile_levels, key=lambda x: abs(x - 0.9))
            
            if str(lower_q) in forecast.columns and str(upper_q) in forecast.columns:
                lower_series = forecast.loc[item_id, str(lower_q)]
                upper_series = forecast.loc[item_id, str(upper_q)]
                
                ax.fill_between(
                    forecast_series.index,
                    lower_series.values,
                    upper_series.values,
                    color="red",
                    alpha=0.2,
                    label="80% Prediction Interval" if item_id == item_ids[0] else "",
                )
    
    ax.set_title("AutoGluon Time Series Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, config.output_dir / "autogluon_forecast.png", dpi=300)
    plt.close(fig)
    print(f" Plot saved -> {config.output_dir / 'autogluon_forecast.png'}")
    
    print("\n AutoGluon forecasting complete")


if __name__ == "__main__":
    main()
