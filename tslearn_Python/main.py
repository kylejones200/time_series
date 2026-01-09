#!/usr/bin/env python3
"""
tslearn for Time Series Machine Learning
Machine learning algorithms specifically designed for time series data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import dtw
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split


def run_clustering(data_scaled, config: dict, script_dir: Path):
    """Run clustering analysis."""
    print("\nFitting TimeSeriesKMeans model...")
    model = TimeSeriesKMeans(
        n_clusters=config["model"]["n_clusters"],
        metric=config["model"]["metric"],
        max_iter=config["model"]["max_iter"],
        random_state=42,
    )
    
    labels = model.fit_predict(data_scaled)
    
    silhouette = silhouette_score(data_scaled.reshape(len(data_scaled), -1), labels)
    print(f"\nClustering Results:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Number of clusters: {config['model']['n_clusters']}")
    
    return model, labels


def run_classification(X_train, X_test, y_train, y_test, config: dict):
    """Run KNN classification with DTW."""
    print("\nFitting KNN classifier with DTW...")
    knn = KNeighborsTimeSeriesClassifier(
        n_neighbors=config["model"].get("n_neighbors", 3),
        metric="dtw",
    )
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nClassification Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Number of neighbors: {config['model'].get('n_neighbors', 3)}")
    
    return knn, y_pred, accuracy


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    task = config["model"].get("task", "clustering")  # "clustering" or "classification"
    
    if task == "classification":
        # For classification, we need labeled data
        # This is a simplified example - in practice, you'd load labeled time series
        print("KNN Classification with DTW requires labeled time series data.")
        print("Creating synthetic example...")
        
        # Create synthetic labeled time series
        np.random.seed(42)
        n_samples = 200
        n_timesteps = 50
        
        X = np.random.rand(n_samples, n_timesteps, 1)
        y = np.random.randint(0, 2, n_samples)  # Binary labels
        
        # Resample to consistent length if needed
        resampler = TimeSeriesResampler(sz=n_timesteps)
        X_resampled = resampler.fit_transform(X)
        
        # Scale
        scaler = TimeSeriesScalerMeanVariance()
        X_scaled = scaler.fit_transform(X_resampled)
        
        # Split
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Run classification
        knn, y_pred, accuracy = run_classification(X_train, X_test, y_train, y_test, config)
        
        # Create visualization
        print("\nCreating visualization...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot sample time series by class
        for class_id in [0, 1]:
            class_mask = y_train == class_id
            class_data = X_train[class_mask]
            
            # Plot a few examples
            for i in range(min(5, len(class_data))):
                axes[0].plot(
                    class_data[i].ravel(),
                    alpha=0.5,
                    label=f"Class {class_id}" if i == 0 else "",
                    color="blue" if class_id == 0 else "red",
                )
        
        axes[0].set_title("Training Time Series by Class")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Value")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot confusion matrix (simplified)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        im = axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        axes[1].figure.colorbar(im, ax=axes[1])
        axes[1].set(xticks=np.arange(2), yticks=np.arange(2),
                   xticklabels=["Class 0", "Class 1"],
                   yticklabels=["Class 0", "Class 1"],
                   title=f"Confusion Matrix (Accuracy: {accuracy:.2f})",
                   ylabel="True Label",
                   xlabel="Predicted Label")
        
        plt.tight_layout()
        
        if config.get("output", {}).get("save_plots", True):
            output_dir = ensure_output_dir(get_output_dir(config, script_dir))
            save_plot(fig, output_dir / "tslearn_classification.png", dpi=300)
            print(f"Plot saved to: {output_dir / 'tslearn_classification.png'}")
        
        print("\n tslearn classification complete")
        
    else:
        # Clustering task
        # Load data using consolidated loader
        series = load_time_series(
            config["data"]["input_file"],
            date_column=config["data"].get("date_col", "date"),
            value_column=config["data"].get("value_col", "value")
        )
        
        print(f"Loaded {len(series)} data points")
        
        # Prepare data for tslearn
        data = series.values.reshape(-1, 1)
        
        # NOTE: For clustering (unsupervised learning), scaling the entire dataset is acceptable
        # since there's no train/test split. However, if this code is adapted for prediction,
        # the data should be split first, then the scaler should be fit on training data only.
        print("\nScaling data for tslearn...")
        scaler = TimeSeriesScalerMeanVariance()
        data_scaled = scaler.fit_transform(to_time_series_dataset(data))
        
        # Run clustering
        model, labels = run_clustering(data_scaled, config, script_dir)
        
        # Create visualization
        print("\nCreating visualization...")
        fig, ax = plt.subplots(figsize=tuple(config.get("plotting", {}).get("figure_size", [12, 6])))
        
        colors = plt.cm.tab10(np.linspace(0, 1, config["model"]["n_clusters"]))
        
        for cluster_id in range(config["model"]["n_clusters"]):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask.flatten()]
            cluster_dates = series.index[cluster_mask.flatten()]
            
            ax.plot(
                cluster_dates,
                cluster_data,
                "o",
                color=colors[cluster_id],
                markersize=config.get("plotting", {}).get("markersize", 4),
                alpha=config.get("plotting", {}).get("alpha", 0.8),
                label=f"Cluster {cluster_id + 1}",
            )
        
        ax.set_title(config.get("plot_titles", {}).get("tslearn_clustering", "tslearn Time Series Clustering"))
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if config.get("output", {}).get("save_plots", True):
            output_dir = ensure_output_dir(get_output_dir(config, script_dir))
            save_plot(fig, output_dir / "tslearn_clustering.png", dpi=300)
            print(f"Plot saved to: {output_dir / 'tslearn_clustering.png'}")
        
        print("\n tslearn clustering complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
