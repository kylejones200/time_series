# AutoGluon for Time Series Forecasting

Automated time series forecasting with AutoGluon's TimeSeriesPredictor using multiple models and automatic hyperparameter tuning.

## Features

- ✅ Automated model selection and ensemble
- ✅ Multiple forecasting models (LightGBM, Prophet, ARIMA, etc.)
- ✅ Automatic hyperparameter tuning
- ✅ Support for univariate and multivariate time series
- ✅ Multiple evaluation metrics

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

- **prediction_length**: Number of steps to forecast
- **freq**: Time series frequency (`D`, `H`, `M`, etc.)
- **eval_metric**: Evaluation metric (`MAPE`, `sMAPE`, `MASE`, `MSE`, `RMSE`)
- **presets**: Quality/speed tradeoff (`best_quality`, `fast_training`, etc.)
- **hyperparameters**: Model-specific hyperparameters

## Model Presets

- **best_quality**: Best accuracy, slower training
- **high_quality**: High accuracy, moderate training time
- **good_quality**: Good accuracy, faster training
- **medium_quality**: Balanced accuracy and speed
- **fast_training**: Fast training, lower accuracy

## Supported Models

AutoGluon automatically selects from:
- LightGBM
- Prophet
- ARIMA
- ETS (Exponential Smoothing)
- Seasonal Naive
- Temporal Fusion Transformer
- And more...

## Outputs

- `outputs/autogluon_forecast.png`: Forecast visualization
- `autogluon_model/`: Saved model directory

## Notes

- AutoGluon automatically handles model selection and ensemble
- Models are saved and can be loaded for future predictions
- Supports both univariate and multivariate time series (via item_id)

