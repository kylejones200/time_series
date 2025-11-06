# BERT: Time Series Classification

Using BERT for time series classification by tokenizing numerical sequences.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path, feature columns, target column
- **Model**: BERT model name, training parameters, number of labels
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Features

- Tokenizes time series as text sequences
- Fine-tunes BERT for classification
- Supports multi-class classification
- Comprehensive evaluation metrics

## Outputs

Classification results and plots saved to `outputs/` directory.

