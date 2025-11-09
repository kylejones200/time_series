"""Sentiment aggregation for war-related corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from textblob import TextBlob

from . import dataset_loaders, preprocessing


@dataclass
class WarSentimentConfig:
    dataset: dataset_loaders.DatasetConfig
    date_col: str = "date"
    text_col: str = "text"
    term_col: str = "term"
    time_format: str | None = None
    aggregate: Literal["mean", "median"] = "mean"
    output_dir: Path | None = None


def analyse(config: WarSentimentConfig) -> pd.DataFrame:
    df = dataset_loaders.load_dataset(config.dataset)
    df = preprocessing.normalize_datetime(df, config.date_col, config.time_format)
    df = preprocessing.add_year_column(df, config.date_col)
    df["polarity"] = df[config.text_col].fillna("").astype(str).apply(lambda t: TextBlob(t).sentiment.polarity)
    df["subjectivity"] = df[config.text_col].fillna("").astype(str).apply(lambda t: TextBlob(t).sentiment.subjectivity)

    agg_func = config.aggregate
    grouped = df.groupby([config.term_col, "year"]).agg(
        polarity=("polarity", agg_func),
        subjectivity=("subjectivity", agg_func),
        count=(config.text_col, "count"),
    )
    return grouped.reset_index()
