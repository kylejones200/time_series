# ARIMA: Autoregressive Integrated Moving Average

Classical time series forecasting using ARIMA models.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: Model type (auto/manual), order parameters, forecast horizon
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Model Types

- **auto**: Automatically select optimal ARIMA order using pmdarima
- **manual**: Use specified order (p, d, q)

## Outputs

Forecast plots saved to `outputs/` directory.

