"""Wrappers for causal inference techniques such as Convergent Cross Mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CCMConfig:
    embedding_dim: int = 3
    tau: int = 1
    library_size: int = 100


def convergent_cross_mapping(source: np.ndarray, target: np.ndarray, config: CCMConfig) -> Tuple[float, float]:
    """Placeholder CCM implementation returning score and p-value.

    Replace this stub with a call into a library such as ``pyEDM`` or
    ``tigramite`` once dependencies are finalised.
    """

    if source.shape != target.shape:
        raise ValueError("Source and target series must be the same length")

    # TODO: integrate real CCM library
    score = float(np.corrcoef(source, target)[0, 1])
    p_value = 1.0  # placeholder
    return score, p_value
