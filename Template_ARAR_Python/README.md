# ARAR: Autoregressive Autoregressive

Time series forecasting using reduced lag sets (ARAR algorithm).

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: Lag selection method (powers_of_2, custom, auto), differencing, forecast horizon
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Features

- Reduced lag set selection
- Automatic differencing
- Flexible lag selection methods

## Outputs

Forecast plots saved to `outputs/` directory.

