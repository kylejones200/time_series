"""Pre-processing helpers for sentiment and linguistic datasets."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def normalize_year_column(df: pd.DataFrame, column: str = "year") -> pd.DataFrame:
    """Ensure the year column is integer typed and sorted."""

    df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    df = df.dropna(subset=[column]).sort_values(column)
    return df.reset_index(drop=True)


def smooth_series(
    series: pd.Series,
    window: int = 5,
    center: bool = True,
) -> pd.Series:
    """Return a rolling mean smoothed version of ``series``."""

    return series.rolling(window=window, center=center, min_periods=1).mean()


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df
