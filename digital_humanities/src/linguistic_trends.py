"""Analysis helpers for n-gram frequency and linguistic change over time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from . import preprocessing


@dataclass
class NgramConfig:
    path: Path
    date_col: str = "year"
    value_col: str = "frequency"
    smoothing_window: Optional[int] = 3


def load_ngram(config: NgramConfig) -> pd.DataFrame:
    df = pd.read_csv(config.path)
    df = preprocessing.ensure_columns(df, [config.date_col, config.value_col])
    df = preprocessing.normalize_year_column(df, column=config.date_col)
    if config.smoothing_window:
        df[f"{config.value_col}_smoothed"] = preprocessing.smooth_series(
            df[config.value_col], window=config.smoothing_window
        )
    return df
