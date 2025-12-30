#!/usr/bin/env python3
"""
MFLEs: Multi-Frequency Learning Ensemble
Ensemble forecasting using multiple frequency components.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Apply SignalPlot's clean defaults
signalplot.apply()


def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj



ts_utils = repo_import("utils.ts_utils")
load_ts_data = ts_utils.load_ts_data
ensure_datetime_index = ts_utils.ensure_datetime_index
split_ts = ts_utils.split_ts
create_time_features = ts_utils.create_time_features
create_lags = ts_utils.create_lags
create_rolling_features = ts_utils.create_rolling_features


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_frequency_components(data, config):
    """Extract features from multiple frequency components."""
    features = create_time_features(data)
    features = create_lags(data, config["model"].get("lags", [1, 2, 3, 7, 14]))
    features = create_rolling_features(
        data,
        windows=config["model"].get("windows", [7, 14, 30]),
        functions=config["model"].get("rolling_funcs", ["mean", "std"]),
    )

    return features


def create_ensemble_model(config):
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
                (
                    "rf",
                    RandomForestRegressor(n_estimators=50, random_state=random_state),
                ),
                (
                    "gbr",
                    GradientBoostingRegressor(
                        n_estimators=50, random_state=random_state
                    ),
                ),
                ("svr", SVR(kernel="rbf")),
            ],
            final_estimator=Ridge(),
            cv=config["model"].get("cv_folds", 5),
        ),
    }

    return model_map.get(model_type, model_map["bagging"])()


def fit_and_predict(model, X_train, y_train, X_test, config):
    """Fit model and generate predictions."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def create_visualizations(data, train_data, test_data, predictions, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=tuple(config["plotting"]["figure_size"]))
    
    ax.plot(
        train_data.index,
        train_data.values,
        "k-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label="Train",
    )

    if test_data is not None:
        ax.plot(
            test_data.index,
            test_data.values,
            "g-",
            linewidth=config["plotting"]["linewidth"],
            alpha=config["plotting"]["alpha"],
            label="Test",
        )

    forecast_index = (
        test_data.index
        if test_data is not None
        else pd.date_range(
            start=train_data.index[-1] + pd.Timedelta(days=1),
            periods=len(predictions),
            freq="D",
        )
    )

    ax.plot(
        forecast_index,
        predictions,
        "r--",
        linewidth=config["plotting"]["linewidth"],
        label="Forecast",
    )

    ax.set_title(config["plot_titles"]["mfles_forecast"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    output_path = output_dir / "mfles_forecast.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def main():
    """Main execution function."""
    config = load_config()

    df = load_ts_data(
        data_path=Path(__file__).parent.parent / "data" / config["data"]["input_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["value_col"],
    )
    df = ensure_datetime_index(df, time_col=config["data"]["date_col"])

    train_df, test_df = split_ts(df, test_size=config["data"]["test_size"])

    features_train = extract_frequency_components(
        train_df[config["data"]["value_col"]], config
    )
    features_test = extract_frequency_components(
        test_df[config["data"]["value_col"]], config
    )

    features_train = features_train.dropna()
    features_test = features_test.dropna()

    X_train = features_train.drop(config["data"]["value_col"], axis=1)
    y_train = features_train[config["data"]["value_col"]]
    X_test = features_test.drop(config["data"]["value_col"], axis=1)
    y_test = features_test[config["data"]["value_col"]]

    model = create_ensemble_model(config)
    predictions = fit_and_predict(model, X_train, y_train, X_test, config)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\nModel Evaluation ({config['model']['ensemble_type']}):")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    create_visualizations(
        df,
        train_df[config["data"]["value_col"]],
        test_df[config["data"]["value_col"]],
        predictions,
        config,
    )

    print(f"✓ MFLEs {config['model']['ensemble_type']} ensemble forecasting complete")


if __name__ == "__main__":
    main()
