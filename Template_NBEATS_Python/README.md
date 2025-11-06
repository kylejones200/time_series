# N-BEATS: Neural Basis Expansion Analysis

Deep learning time series forecasting with interpretable basis expansion.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: Network architecture (stacks, blocks, layers), training epochs, forecast horizon
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Features

- Interpretable neural forecasting
- Automatic basis expansion
- Handles complex patterns

## Outputs

Forecast plots saved to `outputs/` directory.

