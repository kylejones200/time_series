# Bollinger Bands for Time Series Analysis

Technical indicator using moving averages and standard deviations to identify volatility and potential reversal points.

## Features

- ✅ Moving average calculation
- ✅ Upper and lower bands based on standard deviation
- ✅ Volatility visualization
- ✅ Simple and effective technical analysis

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

- **window**: Moving average window size (default: 20)
- **num_std**: Number of standard deviations for bands (default: 2)

## Interpretation

- **Upper Band**: Price may be overbought
- **Lower Band**: Price may be oversold
- **Band Width**: Indicates volatility
- **Price touching bands**: Potential reversal signals

## Outputs

- `outputs/bollinger_bands.png`: Bollinger Bands visualization

