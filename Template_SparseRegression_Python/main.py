#!/usr/bin/env python3
"""Feature importance and supervised-learning visuals aligned with the 2025-11-08 article assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import signalplot
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Apply SignalPlot's clean defaults
signalplot.apply()


@dataclass
class Config:
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


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg["data"]["input_file"]
    output_dir = Path(__file__).parent / cfg["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    return Config(
        data_path=data_path,
        date_col=cfg["data"]["date_col"],
        value_col=cfg["data"]["value_col"],
        season=int(cfg["feature_importance"]["season"]),
        n_splits=int(cfg["feature_importance"]["n_splits"]),
        window_size=int(cfg["sliding_window"]["window_size"]),
        output_dir=output_dir,
        rf_plot=output_dir / cfg["output"]["random_forest_plot"],
        supervised_forecast_plot=output_dir / cfg["output"]["supervised_forecast"],
        supervised_importance_plot=output_dir / cfg["output"]["supervised_importance"],
        seasonal_pattern_plot=output_dir / cfg["output"]["seasonal_pattern"],
        fuel_mix_plot=output_dir / cfg["output"]["fuel_mix_plot"],
    )


def load_dataframe(config: Config) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")

    df = pd.read_csv(config.data_path, skiprows=4)
    df = df.rename(columns=lambda c: c.strip().lower())
    if config.date_col.lower() not in df.columns:
        raise ValueError("Date column not present in CSV")
    df[config.date_col.lower()] = pd.to_datetime(
        df[config.date_col.lower()], errors="coerce"
    )
    df = df.dropna(subset=[config.date_col.lower()])
    return df.sort_values(config.date_col.lower()).reset_index(drop=True)


def build_features(series: pd.Series, season: int) -> pd.DataFrame:
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
    df = df.set_index(config.date_col.lower())
    target = df[config.value_col.lower()].astype(float)
    features_df = build_features(target, config.season)

    features = features_df.columns.drop("y")
    X = features_df[features].values
    y = features_df["y"].values
    idx = np.arange(len(features_df))

    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    importances = np.zeros(len(features), dtype=float)
    maes = []

    for train_idx, test_idx in splitter.split(idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        importances += model.feature_importances_
        maes.append(mean_absolute_error(y_test, preds))

    importances /= max(1, splitter.get_n_splits())
    importance_series = pd.Series(importances, index=features).sort_values(
        ascending=False
    )
    print(f"Random forest baseline MAE: {np.mean(maes):.3f}")

    plt.figure(figsize=(10, 5))
    importance_series.head(15)[::-1].plot(kind="barh")
    plt.title("Top feature importances (random forest)")
    plt.tight_layout()
    plt.savefig(config.rf_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Feature importance plot saved -> {config.rf_plot}")


def create_sliding_window(
    series: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return np.asarray(X), np.asarray(y)


def supervised_learning_pipeline(df: pd.DataFrame, config: Config) -> None:
    df = df.copy()
    date_col = config.date_col.lower()
    value_col = config.value_col.lower()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    generation = df[value_col].values

    X, y = create_sliding_window(generation, config.window_size)
    target_dates = df[date_col].iloc[config.window_size :].reset_index(drop=True)

    train_mask = target_dates < pd.Timestamp("2021-01-01")
    X_train, X_test = X[train_mask.values], X[~train_mask.values]
    y_train, y_test = y[train_mask.values], y[~train_mask.values]
    test_dates = target_dates[~train_mask.values]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    print(
        f"Train RMSE: {np.sqrt(mean_squared_error(y_train, model.predict(X_train))):,.0f}"
    )
    print(f"Test RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred_test)):,.0f}")
    print(f"Test MAE:   {mean_absolute_error(y_test, y_pred_test):,.0f}")
    print(f"Test R²:    {r2_score(y_test, y_pred_test):.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(test_dates, y_test, label="Actual", linewidth=2, alpha=0.8)
    axes[0].plot(
        test_dates,
        y_pred_test,
        label="Predicted",
        linewidth=2,
        alpha=0.8,
        linestyle="--",
    )
    axes[0].set_title("US Electricity Generation Forecast (2021-2025)")
    axes[0].set_ylabel("Generation (thousand MWh)")
    axes[0].legend()
    residuals = y_test - y_pred_test
    axes[1].scatter(y_pred_test, residuals, alpha=0.6)
    axes[1].axhline(0, color="r", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Predicted (thousand MWh)")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residual Plot")
    fig.tight_layout()
    fig.savefig(config.supervised_forecast_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Supervised forecast plot saved -> {config.supervised_forecast_plot}")

    importances = model.feature_importances_
    feature_names = [f"T-{config.window_size - i}" for i in range(config.window_size)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names, importances)
    ax.set_xlabel("Importance")
    ax.set_title("Lag importance (sliding window)")
    fig.tight_layout()
    fig.savefig(config.supervised_importance_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Sliding window feature importance -> {config.supervised_importance_plot}")

    df["month"] = df[date_col].dt.month
    monthly_avg = df.groupby("month")[value_col].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    ax.bar(range(1, 13), monthly_avg.values, alpha=0.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_ylabel("Average Generation (thousand MWh)")
    ax.set_title("Average US Electricity Generation by Month (2001-2025)")
    fig.tight_layout()
    fig.savefig(config.seasonal_pattern_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Seasonal pattern plot saved -> {config.seasonal_pattern_plot}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for fuel in [
        "coal thousand megawatthours",
        "natural gas thousand megawatthours",
        "nuclear thousand megawatthours",
        "conventional hydroelectric thousand megawatthours",
    ]:
        col = fuel.strip().lower()
        if col in df.columns:
            ax.plot(
                df[date_col],
                pd.to_numeric(df[col], errors="coerce"),
                label=fuel.split()[0].title(),
                alpha=0.8,
            )
    ax.set_ylabel("Generation (thousand MWh)")
    ax.set_title("US Electricity Generation by Fuel Source (2001-2025)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.fuel_mix_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Fuel mix plot saved -> {config.fuel_mix_plot}")


def main() -> None:
    config = load_config()
    df = load_dataframe(config)
    feature_importance_pipeline(
        df[[config.date_col.lower(), config.value_col.lower()]], config
    )
    supervised_learning_pipeline(df, config)


if __name__ == "__main__":
    main()
