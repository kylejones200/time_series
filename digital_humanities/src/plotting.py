"""Shared plotting helpers for the Digital Humanities toolkit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_trend(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
    title: str,
    smoothed_col: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[x], df[y], label=y, color="steelblue", alpha=0.6)
    if smoothed_col and smoothed_col in df.columns:
        ax.plot(df[x], df[smoothed_col], label=smoothed_col, color="crimson", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
