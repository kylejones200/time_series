#!/usr/bin/env python3
"""
TSFresh: Time Series Feature Extraction
Automated feature extraction from time series data.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.ensemble import RandomForestClassifier


def prepare_data(data: pd.Series, config: dict) -> pd.DataFrame:
    """Prepare data for TSFresh (requires 'id' and 'time' columns)."""
    df = pd.DataFrame()
    
    n_samples = config["model"].get("n_samples", 10)
    sample_length = len(data) // n_samples
    
    for i in range(n_samples):
        start_idx = i * sample_length
        end_idx = (i + 1) * sample_length if i < n_samples - 1 else len(data)
        
        sample_df = pd.DataFrame(
            {
                "id": i,
                "time": range(len(data.iloc[start_idx:end_idx])),
                "value": data.iloc[start_idx:end_idx].values,
            }
        )
        df = pd.concat([df, sample_df], ignore_index=True)
    
    return df


def extract_tsfresh_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Extract features using TSFresh."""
    extracted_features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=config["model"].get("fc_parameters", "comprehensive"),
        impute_function=impute,
    )
    return extracted_features


def select_relevant_features(X: pd.DataFrame, y: np.ndarray, config: dict) -> pd.DataFrame:
    """Select relevant features."""
    if config["model"].get("feature_selection", True):
        return select_features(X, y)
    return X


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data using consolidated loader
    data = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value")
    )
    
    print(f"Loaded {len(data)} data points")
    
    # Prepare data for TSFresh
    df = prepare_data(data, config)
    print(f"Prepared {len(df)} samples for feature extraction")
    
    # Extract features
    print("\nExtracting TSFresh features...")
    extracted_features = extract_tsfresh_features(df, config)
    print(f"Extracted {len(extracted_features.columns)} features")
    
    # Generate labels (for demonstration - in practice, use real labels)
    labels = np.random.randint(0, 2, size=config["model"].get("n_samples", 10))
    
    # Split data first to avoid leakage
    test_size = config["model"].get("test_size", 0.2)
    n_samples = config["model"].get("n_samples", 10)
    n_train = int(n_samples * (1 - test_size))
    
    train_ids = list(range(n_train))
    test_ids = list(range(n_train, n_samples))
    
    # Split features and labels
    X_train_full = extracted_features.loc[train_ids]
    X_test_full = extracted_features.loc[test_ids]
    y_train = labels[train_ids]
    y_test = labels[test_ids]
    
    # Perform feature selection ONLY on training data
    print("\nSelecting relevant features...")
    X_train = select_relevant_features(X_train_full, y_train, config)
    
    # Apply same feature selection to test set
    if config["model"].get("feature_selection", True):
        selected_cols = X_train.columns
        X_test = X_test_full[selected_cols]
    else:
        X_test = X_test_full
    
    print(f"Selected {len(X_train.columns)} features")
    
    # Train model on selected features
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    feature_importance = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    
    # Create visualization
    print("\nCreating visualization...")
    top_features = feature_importance.head(20)
    
    fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))
    
    ax.barh(
        range(len(top_features)),
        top_features.values,
        color=config.get("plotting", {}).get("style", {}).get("colors", {}).get("primary", "k"),
        alpha=config.get("plotting", {}).get("style", {}).get("alpha", 0.8),
    )
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 20 TSFresh Features by Importance")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot using consolidated utility
    if config.get("output", {}).get("save_plots", True):
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "tsfresh_features.png", dpi=300)
        print(f"Plot saved to: {output_dir / 'tsfresh_features.png'}")
    
    # Save feature importance
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    feature_importance.to_csv(output_dir / "feature_importance.csv", encoding="utf-8")
    print(f"Feature importance saved to: {output_dir / 'feature_importance.csv'}")
    
    print("\n TSFresh feature extraction complete")
    print(f"Extracted {len(X_train.columns)} features")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
