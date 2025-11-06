# Regime Switching Models for Time Series

Markov switching models for time series with structural breaks and regime changes.

## Features

- ✅ Markov switching regression
- ✅ Multiple regimes
- ✅ Switching variance
- ✅ Smoothed regime probabilities
- ✅ Transition matrix

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

- **k_regimes**: Number of regimes (default: 2)
- **trend**: Trend specification (`c` for constant, `n` for none)
- **switching_variance**: Allow variance to switch between regimes

## Applications

- Structural breaks in time series
- Business cycle analysis
- Volatility regimes
- Economic state transitions

## Outputs

- `outputs/regime_switching.png`: Time series with regime probabilities
- Console output: Model summary and transition matrix

