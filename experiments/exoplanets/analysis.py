#!/usr/bin/env python3
"""Exoplanet light curve analysis experiments.

This script consolidates the exploratory, time-series conversion, and PCA
workflow that previously lived across several notebooks (`exo conversion.ipynb`,
`exoplanet pca.ipynb`, and `Exoplanet_Analysis_Complete.ipynb`).

It assumes two CSV files (`exoTrain.csv`, `exoTest.csv`) are located in the
`data/exoplanets/` directory relative to the repository root. These are the
canonical Kepler light-curve datasets published alongside the Kaggle
`NASA Exoplanet Search` competition.

Outputs are written to `experiments/exoplanets/outputs/`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


@dataclass
class Paths:
    repo_root: Path
    data_dir: Path
    train_csv: Path
    test_csv: Path
    output_dir: Path


def build_paths(repo_root: Optional[Path] = None) -> Paths:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "exoplanets"
    train_csv = data_dir / "exoTrain.csv"
    test_csv = data_dir / "exoTest.csv"
    output_dir = repo_root / "experiments" / "exoplanets" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return Paths(repo_root, data_dir, train_csv, test_csv, output_dir)


# ---------------------------------------------------------------------------
# Data loading and inspection
# ---------------------------------------------------------------------------


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected dataset at {csv_path}. Please download the Kaggle "
            "NASA Exoplanet Search data (exoTrain.csv & exoTest.csv) and "
            f"place it in {csv_path.parent}"
        )
    df = pd.read_csv(csv_path)
    return df


def summarize_dataset(df: pd.DataFrame, name: str) -> None:
    flux_cols = [c for c in df.columns if c != "LABEL"]
    print(f"\n{name} summary")
    print("-" * (len(name) + 8))
    print(f"Shape: {df.shape}")
    print("Label distribution:\n" + str(df["LABEL"].value_counts()))
    print(f"First five flux columns statistics:\n{df[flux_cols[:5]].describe()}\n")


# ---------------------------------------------------------------------------
# Time-series conversion and resampling
# ---------------------------------------------------------------------------


def convert_to_weekly_timeseries(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    flux_cols = [c for c in df.columns if c != "LABEL"]

    # Add deterministic UUIDs for reproducibility
    uuids = [f"star_{i:05d}" for i in range(len(df))]
    df_ts = df.copy()
    df_ts["uuid"] = uuids

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_ts[flux_cols])
    df_scaled = pd.DataFrame(scaled, columns=flux_cols)
    df_scaled["uuid"] = uuids

    df_long = df_scaled.melt(id_vars="uuid", var_name="flux_index", value_name="flux")
    df_long["flux_index"] = (
        df_long["flux_index"].str.replace("FLUX.", "", regex=False).astype(int)
    )

    # Create artificial daily timeline
    start_date = pd.Timestamp("2000-01-01")
    df_long["date"] = start_date + pd.to_timedelta(df_long["flux_index"], unit="D")

    df_pivot = df_long.pivot(index="date", columns="uuid", values="flux").sort_index()
    df_weekly = df_pivot.resample("W").mean()

    df_weekly_export = df_weekly.T
    df_weekly_export["LABEL"] = df["LABEL"].values

    output_path = output_dir / "exoplanet_weekly_timeseries.csv"
    df_weekly_export.to_csv(output_path)
    print(f"✓ Weekly time series exported -> {output_path}")

    # Create illustrative plot for the first few stars
    sample_uuids = uuids[:5]
    df_sample = df_long[df_long["uuid"].isin(sample_uuids)]

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df_sample, x="flux_index", y="flux", hue="uuid")
    plt.title("Sample light curves (standardised)", fontsize=14)
    plt.xlabel("Flux measurement index")
    plt.ylabel("Scaled flux")
    plt.tight_layout()
    plot_path = output_dir / "sample_light_curves.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✓ Sample light curve plot saved -> {plot_path}")

    return df_weekly_export


# ---------------------------------------------------------------------------
# PCA pipeline
# ---------------------------------------------------------------------------


def perform_pca(df: pd.DataFrame, n_components: int, output_dir: Path) -> pd.DataFrame:
    flux_cols = [c for c in df.columns if c != "LABEL"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[flux_cols])

    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()

    pca_df = pd.DataFrame(
        components, columns=[f"PC{i+1}" for i in range(components.shape[1])]
    )
    pca_df["LABEL"] = df["LABEL"].values

    output_path = output_dir / "exoplanet_pca_components.csv"
    pca_df.to_csv(output_path, index=False)
    print(f"✓ PCA components exported -> {output_path}")
    print(f"  Total variance explained: {explained:.4f}")

    return pca_df


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run(paths: Paths, pca_components: int = 200) -> None:
    print("=== Exoplanet analysis experiments ===")
    print(f"Data directory: {paths.data_dir}")

    df_train = load_dataset(paths.train_csv)
    summarize_dataset(df_train, "Training set")

    try:
        df_test = load_dataset(paths.test_csv)
        summarize_dataset(df_test, "Test set")
    except FileNotFoundError as err:
        print(f"! Test dataset not found: {err}")

    weekly_df = convert_to_weekly_timeseries(df_train, paths.output_dir)
    perform_pca(df_train, pca_components, paths.output_dir)

    print("=== Experiments complete ===")


if __name__ == "__main__":
    run(build_paths())
