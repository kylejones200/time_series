#!/usr/bin/env python3
"""
MFLEs: Multi-Frequency Learning Ensemble
Ensemble forecasting using multiple frequency components.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def extract_frequency_components(data: pd.Series, config: dict):
    """Extract features from multiple frequency components."""
    from utils.ts_utils import create_time_features, create_lags, create_rolling_features

    time_feats = create_time_features(data)
    lag_feats = create_lags(data, config["model"].get("lags", [1, 2, 3, 7, 14]))
    roll_feats = create_rolling_features(
        data,
        windows=config["model"].get("windows", [7, 14, 30]),
        functions=config["model"].get("rolling_funcs", ["mean", "std"]),
    )

    features = pd.concat([time_feats, lag_feats, roll_feats], axis=1)
    # Prevent leakage: remove the contemporaneous target from features.
    features = features.loc[:, ~features.columns.duplicated()]
    if "value" in features.columns:
        features = features.drop(columns=["value"])

    return features


def create_ensemble_model(config: dict):
    """Create ensemble model based on type."""
    model_type = config["model"]["ensemble_type"]
    random_state = config["model"].get("random_state", 42)
    
    model_map = {
        "bagging": lambda: RandomForestRegressor(
            n_estimators=config["model"].get("n_estimators", 100),
            max_depth=config["model"].get("max_depth", 10),
            random_state=random_state,
        ),
        "boosting": lambda: XGBRegressor(
            n_estimators=config["model"].get("n_estimators", 100),
            learning_rate=config["model"].get("learning_rate", 0.1),
            max_depth=config["model"].get("max_depth", 6),
            random_state=random_state,
        ),
        "stacking": lambda: StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(n_estimators=50, random_state=random_state)),
                ("gbr", GradientBoostingRegressor(n_estimators=50, random_state=random_state)),
                ("svr", SVR(kernel="rbf")),
            ],
            final_estimator=Ridge(),
        ),
    }
    
    return model_map.get(model_type, model_map["bagging"])()


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
    
    print(f"Loaded {len(series)} data points")
    
    # Extract features
    print("\nExtracting multi-frequency features...")
    features = extract_frequency_components(series, config)
    print(f"Extracted {len(features.columns)} features")
    
    # Prepare data - shift target for next-step prediction
    features = features.dropna()
    target = series.shift(-1).loc[features.index]
    valid = ~target.isna()
    features = features.loc[valid]
    target = target.loc[valid]
    
    # Split train/test using consolidated evaluator
    evaluator = Evaluator(test_size=config.get("evaluation", {}).get("test_size", 0.2))
    # Need to split based on feature availability
    split_idx = int(len(features) * (1 - evaluator.test_size))
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
    
    print(f"\nTrain: {len(X_train)} points, Test: {len(X_test)} points")
    
    # Create and fit ensemble model
    ensemble_type = config["model"]["ensemble_type"]
    print(f"\nCreating {ensemble_type} ensemble model...")
    model = create_ensemble_model(config)
    
    print("Training ensemble model...")
    model.fit(X_train, y_train)
    
    # Generate predictions
    print("Generating predictions...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Evaluate
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\nEvaluation Metrics:")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Create visualization
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))
    
    ax.plot(y_train.index[-50:] if len(y_train) > 50 else y_train.index, 
            y_train.values[-50:] if len(y_train) > 50 else y_train.values,
            "k-", lw=1.5, label="Historical (Train)", alpha=0.8)
    ax.plot(y_test.index, y_test.values, "b-", lw=1.5, label="Actual (Test)", alpha=0.8)
    ax.plot(y_test.index, test_pred, "r--", lw=2.0, label=f"{ensemble_type.capitalize()} Forecast", alpha=0.8)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"MFLEs {ensemble_type.capitalize()} Ensemble Forecast (Test MAE: {test_mae:.4f})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / "mfles_ensemble_forecast.png", dpi=300)
    print(f"Plot saved to: {output_dir / 'mfles_ensemble_forecast.png'}")
    
    print("\n MFLEs ensemble forecasting complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
