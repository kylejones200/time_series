"""Preprocessing helpers for war sentiment datasets."""

from __future__ import annotations

import pandas as pd


def normalize_datetime(df: pd.DataFrame, column: str, fmt: str | None = None) -> pd.DataFrame:
    if fmt:
        df[column] = pd.to_datetime(df[column], format=fmt, errors="coerce")
    else:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    df = df.dropna(subset=[column]).sort_values(column)
    return df.reset_index(drop=True)


def add_year_column(df: pd.DataFrame, date_col: str, year_col: str = "year") -> pd.DataFrame:
    df[year_col] = df[date_col].dt.year
    return df
