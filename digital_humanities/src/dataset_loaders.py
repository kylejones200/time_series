"""Utilities for loading and caching digital humanities datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration describing a dataset on disk or reachable via an API."""

    path: Path
    parser: str = "csv"
    encoding: Optional[str] = None
    postprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Load a dataset according to the supplied configuration.

    Parameters
    ----------
    config:
        DatasetConfig describing the input file and any optional post-processing.
    """

    if config.parser == "csv":
        df = pd.read_csv(config.path, encoding=config.encoding)
    elif config.parser == "excel":
        df = pd.read_excel(config.path)
    else:
        raise ValueError(f"Unsupported parser: {config.parser}")

    if config.postprocess is not None:
        df = config.postprocess(df)
    return df


def list_available_datasets(base_dir: Path) -> Dict[str, Path]:
    """Return a mapping of dataset names to paths under ``base_dir``."""

    datasets: Dict[str, Path] = {}
    for path in base_dir.glob("*.csv"):
        datasets[path.stem] = path
    for path in base_dir.glob("*.parquet"):
        datasets[path.stem] = path
    return datasets
