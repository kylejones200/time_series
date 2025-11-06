# Moving Average: Simple Forecasting

Simple moving average and exponential moving average forecasting.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: Method (SMA/EMA/WMA), window size, alpha (for EMA), forecast horizon
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Methods

- **SMA**: Simple Moving Average
- **EMA**: Exponential Moving Average
- **WMA**: Weighted Moving Average

## Outputs

Forecast plots saved to `outputs/` directory.

