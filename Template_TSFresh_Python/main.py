#!/usr/bin/env python3
"""
TSFresh: Time Series Feature Extraction
Automated feature extraction from time series data.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot
from utils.ts_utils import load_ts_data

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(data, config):
    """Prepare data for TSFresh (requires 'id' and 'time' columns)."""
    df = pd.DataFrame()
    
    n_samples = config['model'].get('n_samples', 10)
    sample_length = len(data) // n_samples
    
    for i in range(n_samples):
        start_idx = i * sample_length
        end_idx = (i + 1) * sample_length if i < n_samples - 1 else len(data)
        
        sample_df = pd.DataFrame({
            'id': i,
            'time': range(len(data.iloc[start_idx:end_idx])),
            'value': data.iloc[start_idx:end_idx].values
        })
        df = pd.concat([df, sample_df], ignore_index=True)
    
    return df


def extract_tsfresh_features(df, config):
    """Extract features using TSFresh."""
    extracted_features = extract_features(
        df,
        column_id='id',
        column_sort='time',
        default_fc_parameters=config['model'].get('fc_parameters', 'comprehensive'),
        impute_function=impute
    )
    
    return extracted_features


def select_relevant_features(X, y, config):
    """Select relevant features."""
    if config['model'].get('feature_selection', True):
        return select_features(X, y)
    return X


def create_visualizations(feature_importance, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    top_features = feature_importance.head(20)
    
    fig, ax = setup_figure(config)
    
    ax.barh(range(len(top_features)), top_features.values,
            color=config['plotting']['style']['colors']['primary'],
            alpha=config['plotting']['style']['alpha'])
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'tsfresh_features.png', config)
     for _ in [None] if config['output']['save_plots']]
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    data = load_ts_data(
        Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    
    df = prepare_data(data, config)
    
    labels = np.random.randint(0, 2, size=config['model'].get('n_samples', 10))
    labels_df = pd.DataFrame({'id': range(len(labels)), 'target': labels})
    
    extracted_features = extract_tsfresh_features(df, config)
    X = select_relevant_features(extracted_features, labels, config)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    feature_importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    
    create_visualizations(feature_importance, config)
    
    print("✓ TSFresh feature extraction complete")
    print(f"Extracted {len(X.columns)} features")


if __name__ == "__main__":
    main()

