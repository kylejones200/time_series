# LSTM for Time Series Forecasting

Long Short-Term Memory (LSTM) networks for time series forecasting using TensorFlow/Keras.

## Features

- ✅ Works with any univariate time series
- ✅ Automatic sequence creation with sliding windows
- ✅ Train/validation/test split
- ✅ Proper data scaling for neural networks
- ✅ LSTM, GRU, and Bidirectional variants
- ✅ Early stopping and learning rate reduction
- ✅ Comprehensive evaluation metrics
- ✅ Multi-step ahead forecasting

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file in the shared `data/` directory
2. Update `config.yaml` with your data file name and column names
3. Run:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- **Model type**: `lstm`, `gru`, or `bidirectional_lstm`
- **Sequence length**: Number of past values to use (default: 30)
- **Hidden units**: List of units per layer (default: [64, 32])
- **Dropout**: Regularization rate (default: 0.2)
- **Training**: Epochs, batch size, early stopping patience

## When to Use LSTM

- ✅ Long sequences with complex dependencies
- ✅ Non-linear relationships
- ✅ Plenty of training data (1000+ samples recommended)
- ✅ Complex patterns that simpler models can't capture

## Model Types

- **LSTM**: Standard, handles long-term dependencies well
- **GRU**: Faster, simpler, often performs similarly to LSTM
- **Bidirectional**: Processes sequences forward and backward (can't forecast future!)

## Outputs

- `outputs/lstm_forecast.png`: Test set predictions vs actual
- `outputs/lstm_future_forecast.png`: Historical data, test predictions, and future forecast

