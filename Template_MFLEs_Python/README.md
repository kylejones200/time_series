# MFLEs: Multi-Frequency Learning Ensemble

Ensemble forecasting using multiple frequency components and feature engineering.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: 
  - `ensemble_type`: "bagging", "boosting", or "stacking"
  - Ensemble parameters, lag features, rolling windows
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Features

- ✅ Multi-frequency feature extraction
- ✅ Lag features
- ✅ Rolling window statistics
- ✅ **Bagging**: Random Forest ensemble
- ✅ **Boosting**: XGBoost gradient boosting
- ✅ **Stacking**: Multi-model stacking with meta-learner
- ✅ Model evaluation metrics (MAE, RMSE, R²)
- ✅ Automatic feature engineering

## Outputs

Forecast plots saved to `outputs/` directory.

