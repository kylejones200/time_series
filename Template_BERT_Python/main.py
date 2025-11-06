#!/usr/bin/env python3
"""
BERT: Time Series Classification
Using BERT for time series classification by tokenizing numerical sequences.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import warnings
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(config):
    """Load time series classification data."""
    data_path = Path(__file__).parent.parent / 'data' / config['data']['input_file']
    df = pd.read_csv(data_path)
    
    X_cols = config['data']['feature_cols']
    y_col = config['data']['target_col']
    
    X = df[X_cols].values if isinstance(X_cols, list) else df[[X_cols]].values
    y = df[y_col].values
    
    return X, y


def tokenize_series(series, tokenizer, max_length=128):
    """Tokenize time series as string sequence."""
    series_str = " ".join(map(str, series))
    return tokenizer(series_str, truncation=True, padding="max_length", 
                     max_length=max_length, return_tensors="pt")


def create_dataset(X, y, tokenizer, config):
    """Create PyTorch dataset from time series."""
    tokens = [tokenize_series(X[i], tokenizer, config['model'].get('max_length', 128)) 
              for i in range(len(X))]
    
    class TimeSeriesDataset(torch.utils.data.Dataset):
        def __init__(self, tokens, labels):
            self.tokens = tokens
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.tokens[idx]['input_ids'].squeeze(),
                'attention_mask': self.tokens[idx]['attention_mask'].squeeze(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
    
    return TimeSeriesDataset(tokens, y)


def create_model(config):
    """Create BERT model for classification."""
    num_labels = config['model'].get('num_labels', 2)
    model_name = config['model'].get('model_name', 'bert-base-uncased')
    
    return BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )


def train_model(model, train_dataset, val_dataset, config):
    """Train BERT model."""
    training_args = TrainingArguments(
        output_dir=Path(__file__).parent / 'outputs' / 'bert_model',
        num_train_epochs=config['model'].get('epochs', 3),
        per_device_train_batch_size=config['model'].get('batch_size', 8),
        per_device_eval_batch_size=config['model'].get('batch_size', 8),
        warmup_steps=config['model'].get('warmup_steps', 500),
        weight_decay=config['model'].get('weight_decay', 0.01),
        logging_dir=Path(__file__).parent / 'outputs' / 'logs',
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


def evaluate_model(trainer, test_dataset, config):
    """Evaluate model and return predictions."""
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    accuracy = accuracy_score(true_labels, pred_labels)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))
    
    return pred_labels, true_labels, accuracy


def create_visualizations(y_true, y_pred, accuracy, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = setup_figure(config)
    
    unique_labels = np.unique(y_true)
    colors_map = {
        0: config['plotting']['style']['colors']['primary'],
        1: config['plotting']['style']['colors']['secondary'],
        2: config['plotting']['style']['colors']['accent'],
    }
    
    [[ax.scatter(i, y_true[i], 
                 c=colors_map.get(y_true[i], 'k'),
                 s=config['plotting']['style']['markersize'] * 10,
                 alpha=config['plotting']['style']['alpha'],
                 marker='o' if y_true[i] == y_pred[i] else 'x',
                 label=f'Class {y_true[i]}' if i == 0 or y_true[i] != y_true[i-1] else '')
      for i in range(len(y_true))]
     for _ in [None]]
    
    ax.set_xlabel('Sample')
    ax.set_ylabel('Class')
    ax.set_title(f'Classification Results (Accuracy: {accuracy:.2%})')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'bert_classification.png', config)
     for _ in [None] if config['output']['save_plots']]
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    X, y = load_data(config)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model'].get('test_size', 0.2), 
        random_state=config['model'].get('random_state', 42)
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, 
        random_state=config['model'].get('random_state', 42)
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model'].get('model_name', 'bert-base-uncased')
    )
    
    train_dataset = create_dataset(X_train, y_train, tokenizer, config)
    val_dataset = create_dataset(X_val, y_val, tokenizer, config)
    test_dataset = create_dataset(X_test, y_test, tokenizer, config)
    
    model = create_model(config)
    trainer = train_model(model, train_dataset, val_dataset, config)
    y_pred, y_true, accuracy = evaluate_model(trainer, test_dataset, config)
    create_visualizations(y_true, y_pred, accuracy, config)
    
    print("✓ BERT time series classification complete")


if __name__ == "__main__":
    main()

