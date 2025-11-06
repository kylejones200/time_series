# Time Series Decomposition

Decompose time series into trend, seasonal, and residual components.

## Features

- ✅ Additive and multiplicative decomposition
- ✅ Trend extraction
- ✅ Seasonal component identification
- ✅ Residual analysis
- ✅ Comprehensive visualization

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

- **decomposition_model**: `additive` or `multiplicative`
- **period**: Period for seasonal component (null for auto-detect)

## Decomposition Models

### Additive
- `y(t) = Trend(t) + Seasonal(t) + Residual(t)`
- Best when seasonal variation is constant

### Multiplicative
- `y(t) = Trend(t) * Seasonal(t) * Residual(t)`
- Best when seasonal variation increases with trend

## Components

- **Trend**: Long-term direction
- **Seasonal**: Repeating patterns
- **Residual**: Random variation

## Outputs

- `outputs/time_series_decomposition.png`: Four-panel decomposition plot
- Console output: Component statistics

