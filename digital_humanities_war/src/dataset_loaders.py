"""Dataset loading utilities for war sentiment studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


@dataclass
class DatasetConfig:
    path: Path
    parser: str = "csv"
    encoding: Optional[str] = None
    postprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    if config.parser == "csv":
        df = pd.read_csv(config.path, encoding=config.encoding)
    elif config.parser == "parquet":
        df = pd.read_parquet(config.path)
    else:
        raise ValueError(f"Unsupported parser: {config.parser}")

    if config.postprocess:
        df = config.postprocess(df)
    return df
