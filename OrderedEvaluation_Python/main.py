#!/usr/bin/env python3
"""Ordered time series model evaluation pipeline using consolidated utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, confusion_matrix


@dataclass
class Config:
    """Configuration dataclass for this template."""
    n_points: int
    class_labels: list[int]
    intervene_cost: float
    review_cost: float
    none_cost: float
    risk_levels: list[float]
    intervention_effects: dict[str, float]
    output_dir: Path


def parse_config(config_dict: dict, script_dir: Path) -> Config:
    """Parse config dictionary into Config dataclass."""
    output_dir = ensure_output_dir(Path(script_dir) / "outputs")
    
    return Config(
        n_points=config_dict["simulation"]["n_points"],
        class_labels=config_dict["simulation"]["class_labels"],
        intervene_cost=config_dict["policy"]["intervene_cost"],
        review_cost=config_dict["policy"]["review_cost"],
        none_cost=config_dict["policy"]["none_cost"],
        risk_levels=config_dict["policy"]["risk_levels"],
        intervention_effects=config_dict["policy"]["intervention_effects"],
        output_dir=output_dir,
    )


def simulate_series(config: Config) -> pd.DataFrame:
    """Simulate time series with labels."""
    rng = np.random.default_rng(42)
    labels = config.class_labels
    n = config.n_points
    
    true = [rng.choice(labels)]
    for _ in range(n - 1):
        last = true[-1]
        move = rng.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
        next_val = int(np.clip(last + move, min(labels), max(labels)))
        true.append(next_val)
    
    model_a = [
        int(np.clip(x - rng.choice([0, 1], p=[0.8, 0.2]), min(labels), max(labels)))
        for x in true
    ]
    model_b = [
        int(np.clip(x + rng.choice([0, 1], p=[0.8, 0.2]), min(labels), max(labels)))
        for x in true
    ]
    
    df = pd.DataFrame({"True": true, "Model_A": model_a, "Model_B": model_b})
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute evaluation metrics."""
    kappa_a = cohen_kappa_score(df["True"], df["Model_A"], weights="quadratic")
    kappa_b = cohen_kappa_score(df["True"], df["Model_B"], weights="quadratic")
    
    error_a = (df["True"] - df["Model_A"]).abs()
    error_b = (df["True"] - df["Model_B"]).abs()
    stat, p_value = wilcoxon(error_a, error_b)
    
    return {
        "kappa_a": kappa_a,
        "kappa_b": kappa_b,
        "wilcoxon_stat": stat,
        "wilcoxon_p": p_value,
    }


def create_visualizations(df: pd.DataFrame, metrics: dict, config: Config) -> None:
    """Create evaluation visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Confusion matrices
    cm_a = confusion_matrix(df["True"], df["Model_A"])
    cm_b = confusion_matrix(df["True"], df["Model_B"])
    
    sns.heatmap(cm_a, annot=True, fmt="d", ax=axes[0, 0], cmap="Blues")
    axes[0, 0].set_title(f"Model A Confusion Matrix (κ={metrics['kappa_a']:.3f})")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")
    
    sns.heatmap(cm_b, annot=True, fmt="d", ax=axes[0, 1], cmap="Oranges")
    axes[0, 1].set_title(f"Model B Confusion Matrix (κ={metrics['kappa_b']:.3f})")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Actual")
    
    # Error comparison
    error_a = (df["True"] - df["Model_A"]).abs()
    error_b = (df["True"] - df["Model_B"]).abs()
    
    axes[1, 0].plot(range(len(error_a)), error_a, "b-", alpha=0.6, label="Model A Error")
    axes[1, 0].plot(range(len(error_b)), error_b, "r-", alpha=0.6, label="Model B Error")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Absolute Error")
    axes[1, 0].set_title(f"Error Comparison (Wilcoxon p={metrics['wilcoxon_p']:.4f})")
    axes[1, 0].legend(loc="best")
    axes[1, 0].grid(True, alpha=0.3)
    
    # True vs predictions scatter
    axes[1, 1].scatter(df["True"], df["Model_A"], alpha=0.6, s=20, label="Model A")
    axes[1, 1].scatter(df["True"], df["Model_B"], alpha=0.6, s=20, label="Model B")
    axes[1, 1].plot([df["True"].min(), df["True"].max()], [df["True"].min(), df["True"].max()], "k--", lw=2, label="Perfect")
    axes[1, 1].set_xlabel("True")
    axes[1, 1].set_ylabel("Predicted")
    axes[1, 1].set_title("True vs Predicted")
    axes[1, 1].legend(loc="best")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, config.output_dir / "ordered_evaluation.png", dpi=300)
    plt.close(fig)
    print(f" Evaluation plot saved -> {config.output_dir / 'ordered_evaluation.png'}")


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config_dict = load_config()
    
    # Parse into Config dataclass
    config = parse_config(config_dict, script_dir)
    
    # Simulate series
    print("Simulating time series...")
    df = simulate_series(config)
    print(f"Generated {len(df)} data points")
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics = compute_metrics(df)
    
    print(f"\nEvaluation Metrics:")
    print(f"  Model A Kappa: {metrics['kappa_a']:.4f}")
    print(f"  Model B Kappa: {metrics['kappa_b']:.4f}")
    print(f"  Wilcoxon p-value: {metrics['wilcoxon_p']:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, metrics, config)
    
    print("\n Ordered evaluation complete")


if __name__ == "__main__":
    main()
