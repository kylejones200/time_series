"""Sentiment trend utilities driven by configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from . import dataset_loaders, preprocessing


@dataclass
class SentimentConfig:
    dataset: dataset_loaders.DatasetConfig
    date_col: str
    value_col: str
    smoothing_window: Optional[int] = 5


def load_sentiment(config: SentimentConfig) -> pd.DataFrame:
    """Load and prepare sentiment data according to the config."""

    df = dataset_loaders.load_dataset(config.dataset)
    df = preprocessing.ensure_columns(df, [config.date_col, config.value_col])
    df = preprocessing.normalize_year_column(df, column=config.date_col)
    if config.smoothing_window:
        df[f"{config.value_col}_smoothed"] = preprocessing.smooth_series(
            df[config.value_col], window=config.smoothing_window
        )
    return df


def export_summary(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write a CSV summary of the provided sentiment DataFrame."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sentiment_summary.csv"
    df.describe().to_csv(output_path)
    return output_path
