#!/usr/bin/env python3
"""
TSFresh: Time Series Feature Extraction
Automated feature extraction from time series data.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.ensemble import RandomForestClassifier

# Apply SignalPlot's clean defaults
signalplot.apply()


def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj



ts_utils = repo_import("utils.ts_utils")
load_ts_data = ts_utils.load_ts_data


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(data, config):
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


def extract_tsfresh_features(df, config):
    """Extract features using TSFresh."""
    extracted_features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=config["model"].get("fc_parameters", "comprehensive"),
        impute_function=impute,
    )

    return extracted_features


def select_relevant_features(X, y, config):
    """Select relevant features."""
    if config["model"].get("feature_selection", True):
        return select_features(X, y)
    return X


def create_visualizations(feature_importance, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    top_features = feature_importance.head(20)

    fig, ax = plt.subplots(figsize=config)

    ax.barh(
        range(len(top_features)),
        top_features.values,
        color=config["plotting"]["style"]["colors"]["primary"],
        alpha=config["plotting"]["style"]["alpha"],
    )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()

    [
        fig.savefig(output_dir / "tsfresh_features.png", dpi=300, bbox_inches="tight", facecolor="white")
        for _ in [None]
        if config["output"]["save_plots"]
    ]
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    data = load_ts_data(
        Path(__file__).parent.parent / "data" / config["data"]["input_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["value_col"],
    )

    df = prepare_data(data, config)

    labels = np.random.randint(0, 2, size=config["model"].get("n_samples", 10))

    # Split data first to avoid leakage
    # Use time-aware split (no shuffling) to respect temporal order
    test_size = config["model"].get("test_size", 0.2)
    n_samples = config["model"].get("n_samples", 10)
    n_train = int(n_samples * (1 - test_size))

    train_ids = list(range(n_train))
    # test_ids = list(range(n_train, n_samples))  # Reserved for future evaluation

    # Extract features for all data (needed for feature extraction)
    extracted_features = extract_tsfresh_features(df, config)

    # Split features and labels
    X_train_full = extracted_features.loc[train_ids]
    # X_test_full = extracted_features.loc[test_ids]  # Reserved for future evaluation
    y_train = labels[train_ids]
    # y_test = labels[test_ids]  # Reserved for future evaluation

    # Perform feature selection ONLY on training data
    X_train = select_relevant_features(X_train_full, y_train, config)

    # Apply same feature selection to test set (keep only selected features)
    # Reserved for future evaluation:
    # if config["model"].get("feature_selection", True):
    #     selected_cols = X_train.columns
    #     X_test = X_test_full[selected_cols]
    # else:
    #     X_test = X_test_full

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    feature_importance = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    create_visualizations(feature_importance, config)

    print("✓ TSFresh feature extraction complete")
    print(f"Extracted {len(X_train.columns)} features (after selection)")


if __name__ == "__main__":
    main()
