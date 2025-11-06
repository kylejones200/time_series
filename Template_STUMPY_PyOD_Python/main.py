#!/usr/bin/env python3
"""
STUMPY + PyOD for Anomaly Detection
Matrix profile (STUMPY) and Python Outlier Detection (PyOD) for time series anomaly detection.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
import stumpy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def stumpy_anomaly_detection(data, window_size, percentile_threshold):
    """Detect anomalies using STUMPY matrix profile."""
    mp = stumpy.stump(data.values.flatten(), m=window_size)
    matrix_profile = mp[:, 0]
    threshold = np.percentile(matrix_profile, percentile_threshold)
    anomalies = (matrix_profile > threshold).astype(int)
    return anomalies, matrix_profile


def pyod_anomaly_detection(data, method, contamination):
    """Detect anomalies using PyOD."""
    method_map = {
        'IForest': IForest(contamination=contamination, random_state=42),
        'LOF': LOF(contamination=contamination),
        'OCSVM': OCSVM(contamination=contamination),
    }
    
    model = method_map.get(method, method_map['IForest'])
    model.fit(data.values.reshape(-1, 1))
    predictions = model.predict(data.values.reshape(-1, 1))
    scores = model.decision_scores_
    
    return predictions, scores


def evaluate_anomalies(y_true, y_pred):
    """Evaluate anomaly detection performance."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_anomalies': int(y_pred.sum())
    }


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col=config['data']['date_col'])
    
    results = {}
    
    if config['model']['use_stumpy']:
        anomalies_stumpy, mp = stumpy_anomaly_detection(
            df, config['model']['stumpy_window'], config['model']['stumpy_percentile']
        )
        if config['data'].get('true_anomalies_col'):
            true_anomalies = df[config['data']['true_anomalies_col']].values
            results['STUMPY'] = evaluate_anomalies(true_anomalies, anomalies_stumpy)
        else:
            results['STUMPY'] = {'n_anomalies': int(anomalies_stumpy.sum())}
    
    if config['model']['use_pyod']:
        anomalies_pyod, scores_pyod = pyod_anomaly_detection(
            df, config['model']['pyod_method'], config['model']['pyod_contamination']
        )
        if config['data'].get('true_anomalies_col'):
            true_anomalies = df[config['data']['true_anomalies_col']].values
            results['PyOD'] = evaluate_anomalies(true_anomalies, anomalies_pyod)
        else:
            results['PyOD'] = {'n_anomalies': int(anomalies_pyod.sum())}
    
    print("\nAnomaly Detection Results:")
    print("=" * 70)
    for method, metrics in results.items():
        print(f"\n{method}:")
        [print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
         for k, v in metrics.items()]
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    for ax in axes:
        apply_plot_style(ax, {'plotting': config['plotting']})
    
    axes[0].plot(df.index, df[config['data']['value_col']].values,
                 'k-', linewidth=config['plotting']['linewidth'],
                 alpha=config['plotting']['alpha'], label='Time Series')
    
    if config['model']['use_stumpy']:
        stumpy_anomalies_idx = df.index[anomalies_stumpy == 1]
        axes[0].scatter(stumpy_anomalies_idx, df.loc[stumpy_anomalies_idx, config['data']['value_col']].values,
                        c='r', s=config['plotting']['markersize'] * 20,
                        label='STUMPY Anomalies', zorder=5, alpha=0.7)
    
    if config['model']['use_pyod']:
        pyod_anomalies_idx = df.index[anomalies_pyod == 1]
        axes[0].scatter(pyod_anomalies_idx, df.loc[pyod_anomalies_idx, config['data']['value_col']].values,
                        c='b', marker='x', s=config['plotting']['markersize'] * 30,
                        label='PyOD Anomalies', zorder=5, alpha=0.7)
    
    axes[0].set_title(config['plot_titles']['anomaly_detection'])
    axes[0].set_ylabel('Value')
    apply_legend(axes[0], config['plotting']['legend'])
    
    if config['model']['use_stumpy']:
        axes[1].plot(df.index[:len(mp)], mp,
                     'r-', linewidth=config['plotting']['linewidth'],
                     alpha=config['plotting']['alpha'], label='Matrix Profile')
        threshold = np.percentile(mp, config['model']['stumpy_percentile'])
        axes[1].axhline(y=threshold, color='r', linestyle='--',
                        linewidth=config['plotting']['linewidth'], label='Threshold')
        axes[1].set_title('STUMPY Matrix Profile')
        axes[1].set_ylabel('Matrix Profile')
        apply_legend(axes[1], config['plotting']['legend'])
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "outputs" / "stumpy_pyod_anomalies.png"
    save_plot(fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()

