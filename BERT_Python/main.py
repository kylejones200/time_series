#!/usr/bin/env python3
"""
BERT: Time Series Classification
Using BERT for time series classification by tokenizing numerical sequences.
"""

import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, classification_report

os.environ["WANDB_DISABLED"] = "true"


def load_data(config: dict):
    """Load time series classification data."""
    data_path = Path(__file__).parent.parent / "data" / config["data"]["input_file"]
    df = pd.read_csv(data_path, encoding="utf-8")
    
    X_cols = config["data"]["feature_cols"]
    y_col = config["data"]["target_col"]
    
    X = df[X_cols].values if isinstance(X_cols, list) else df[[X_cols]].values
    y = df[y_col].values
    
    return X, y


def tokenize_series(series: np.ndarray, tokenizer, max_length: int = 128):
    """Tokenize time series as string sequence."""
    series_str = " ".join(map(str, series))
    return tokenizer(
        series_str,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def create_dataset(X: np.ndarray, y: np.ndarray, tokenizer, config: dict):
    """Create PyTorch dataset from time series."""
    tokens = [
        tokenize_series(X[i], tokenizer, config["model"].get("max_length", 128))
        for i in range(len(X))
    ]
    
    class TimeSeriesDataset(torch.utils.data.Dataset):
        def __init__(self, tokens, labels):
            self.tokens = tokens
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return {
                "input_ids": self.tokens[idx]["input_ids"].squeeze(),
                "attention_mask": self.tokens[idx]["attention_mask"].squeeze(),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }
    
    return TimeSeriesDataset(tokens, y)


def create_model(config: dict):
    """Create BERT model for classification."""
    num_labels = config["model"].get("num_labels", 2)
    model_name = config["model"].get("model_name", "bert-base-uncased")
    
    return BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def train_model(model, train_dataset, val_dataset, config: dict, script_dir: Path):
    """Train BERT model."""
    training_args = TrainingArguments(
        output_dir=script_dir / "outputs" / "bert_model",
        num_train_epochs=config["model"].get("epochs", 3),
        per_device_train_batch_size=config["model"].get("batch_size", 8),
        per_device_eval_batch_size=config["model"].get("batch_size", 8),
        warmup_steps=config["model"].get("warmup_steps", 500),
        weight_decay=config["model"].get("weight_decay", 0.01),
        logging_dir=script_dir / "outputs" / "logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    return trainer


def evaluate_model(trainer, test_dataset, config: dict):
    """Evaluate model and return predictions."""
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    accuracy = accuracy_score(true_labels, pred_labels)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))
    
    return pred_labels, true_labels, accuracy


def create_visualizations(y_true: np.ndarray, y_pred: np.ndarray, accuracy: float, config: dict, script_dir: Path):
    """Generate clean visualizations."""
    fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))
    
    colors_map = {
        0: config.get("plotting", {}).get("style", {}).get("colors", {}).get("primary", "k"),
        1: config.get("plotting", {}).get("style", {}).get("colors", {}).get("secondary", "r"),
        2: config.get("plotting", {}).get("style", {}).get("colors", {}).get("accent", "b"),
    }
    
    for i in range(len(y_true)):
        color = colors_map.get(y_true[i], "k")
        marker = "o" if y_true[i] == y_pred[i] else "x"
        ax.scatter(
            i,
            y_true[i],
            c=color,
            s=config.get("plotting", {}).get("style", {}).get("markersize", 4) * 10,
            alpha=config.get("plotting", {}).get("style", {}).get("alpha", 0.8),
            marker=marker,
        )
    
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class")
    ax.set_title(f"BERT Classification Results (Accuracy: {accuracy:.2%})")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if config.get("output", {}).get("save_plots", True):
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "bert_classification.png", dpi=300)
        print(f"Plot saved to: {output_dir / 'bert_classification.png'}")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data
    X, y = load_data(config)
    print(f"Loaded {len(X)} samples with {len(X[0]) if len(X) > 0 else 0} features")
    
    # Perform time-aware splitting
    test_size = config["model"].get("test_size", 0.2)
    val_size = 0.2  # Fixed validation split from training data
    
    # Calculate split indices
    total_samples = len(X)
    test_split_idx = int(total_samples * (1 - test_size))
    val_split_idx = int(test_split_idx * (1 - val_size))
    
    X_train, X_temp = X[:val_split_idx], X[val_split_idx:]
    y_train, y_temp = y[:val_split_idx], y[val_split_idx:]
    
    X_val, X_test = X_temp[: test_split_idx - val_split_idx], X_temp[test_split_idx - val_split_idx :]
    y_val, y_test = y_temp[: test_split_idx - val_split_idx], y_temp[test_split_idx - val_split_idx :]
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"].get("model_name", "bert-base-uncased")
    )
    
    train_dataset = create_dataset(X_train, y_train, tokenizer, config)
    val_dataset = create_dataset(X_val, y_val, tokenizer, config)
    test_dataset = create_dataset(X_test, y_test, tokenizer, config)
    
    # Create and train model
    print("\nCreating BERT model...")
    model = create_model(config)
    
    print("Training model...")
    trainer = train_model(model, train_dataset, val_dataset, config, script_dir)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred, y_true, accuracy = evaluate_model(trainer, test_dataset, config)
    
    # Create visualizations
    print("\nCreating visualization...")
    create_visualizations(y_true, y_pred, accuracy, config, script_dir)
    
    print("\n BERT time series classification complete")


if __name__ == "__main__":
    main()
