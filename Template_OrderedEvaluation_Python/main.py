#!/usr/bin/env python3
"""Ordered time series model evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_plot_style, save_plot


@dataclass
class Config:
    n_points: int
    class_labels: list[int]
    intervene_cost: float
    review_cost: float
    none_cost: float
    risk_levels: list[float]
    intervention_effects: dict[str, float]
    output_dir: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    return Config(
        n_points=cfg['simulation']['n_points'],
        class_labels=cfg['simulation']['class_labels'],
        intervene_cost=cfg['policy']['intervene_cost'],
        review_cost=cfg['policy']['review_cost'],
        none_cost=cfg['policy']['none_cost'],
        risk_levels=cfg['policy']['risk_levels'],
        intervention_effects=cfg['policy']['intervention_effects'],
        output_dir=output_dir,
    )


def simulate_series(config: Config) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    labels = config.class_labels
    n = config.n_points

    true = [rng.choice(labels)]
    for _ in range(n - 1):
        last = true[-1]
        move = rng.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
        next_val = int(np.clip(last + move, min(labels), max(labels)))
        true.append(next_val)

    model_a = [int(np.clip(x - rng.choice([0, 1], p=[0.8, 0.2]), min(labels), max(labels))) for x in true]
    model_b = [int(np.clip(x + rng.choice([0, 1], p=[0.8, 0.2]), min(labels), max(labels))) for x in true]

    df = pd.DataFrame({'True': true, 'Model_A': model_a, 'Model_B': model_b})
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    kappa_a = cohen_kappa_score(df['True'], df['Model_A'], weights='quadratic')
    kappa_b = cohen_kappa_score(df['True'], df['Model_B'], weights='quadratic')

    error_a = (df['True'] - df['Model_A']).abs()
    error_b = (df['True'] - df['Model_B']).abs()
    stat, p_value = wilcoxon(error_a, error_b)

    return {
        'WeightedKappa_ModelA': kappa_a,
        'WeightedKappa_ModelB': kappa_b,
        'Wilcoxon_p': p_value,
    }


def make_calibration_plot(df: pd.DataFrame, config: Config) -> None:
    counts = pd.DataFrame({
        'True': df['True'].value_counts(normalize=True).sort_index(),
        'Model_A': df['Model_A'].value_counts(normalize=True).sort_index(),
        'Model_B': df['Model_B'].value_counts(normalize=True).sort_index(),
    }).fillna(0)

    ax = counts.plot(kind='bar', figsize=(8, 5))
    ax.set_title('Calibration: Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Proportion')
    plt.tight_layout()
    path = config.output_dir / 'calibration.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Calibration plot -> {path}")


def make_confusion_matrices(df: pd.DataFrame, config: Config) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(axes, ['Model_A', 'Model_B'], ['Model A', 'Model B']):
        cm = confusion_matrix(df['True'], df[col], labels=config.class_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f"Confusion Matrix - {title}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    path = config.output_dir / 'confusion_matrices.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Confusion matrices -> {path}")


def policy_decision(prediction: int) -> str:
    if prediction >= 2:
        return 'intervene'
    if prediction == 1:
        return 'review'
    return 'none'


def compute_policy_costs(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    cost_map = {
        'intervene': config.intervene_cost,
        'review': config.review_cost,
        'none': config.none_cost,
    }
    risk_map = dict(zip(config.class_labels, config.risk_levels))
    effect = config.intervention_effects

    results = []
    for model in ['Model_A', 'Model_B']:
        policies = df[model].apply(policy_decision)
        direct_cost = policies.map(cost_map)
        expected_loss = df['True'].map(risk_map) * policies.map(lambda p: effect[p])
        total_cost = direct_cost + expected_loss
        results.append({
            'model': model,
            'expected_cost': total_cost.mean(),
            'direct_cost': direct_cost.mean(),
            'loss': expected_loss.mean(),
        })

    return pd.DataFrame(results)


def main():
    config = load_config()
    df = simulate_series(config)

    metrics = compute_metrics(df)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    make_calibration_plot(df, config)
    make_confusion_matrices(df, config)

    policy_df = compute_policy_costs(df, config)
    policy_path = config.output_dir / 'policy_costs.csv'
    policy_df.to_csv(policy_path, index=False)
    print(f"✓ Policy costs -> {policy_path}")


if __name__ == "__main__":
    main()
