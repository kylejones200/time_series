"""Reusable plotting helpers for war sentiment analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_term_trends(df: pd.DataFrame, output_dir: Path, term_col: str = "term") -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for term, group in df.groupby(term_col):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(group["year"], group["polarity"], marker="o", label="Polarity")
        ax.set_title(f"Sentiment polarity for {term}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean polarity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = output_dir / f"{term}_polarity.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
