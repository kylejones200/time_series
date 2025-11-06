# Nixtla: StatsForecast

Fast statistical forecasting with Nixtla's StatsForecast library.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: Model types (AutoARIMA, AutoETS, AutoTheta, etc.), frequency, forecast horizon
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Model Types

- **AutoARIMA**: Automatic ARIMA selection
- **AutoETS**: Exponential Smoothing
- **AutoTheta**: Theta method
- **AutoCES**: Complex Exponential Smoothing
- **DynamicOptimizedTheta**: Optimized Theta
- **SeasonalNaive**: Baseline seasonal model

## Outputs

Forecast plots saved to `outputs/` directory.

