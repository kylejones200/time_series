#!/usr/bin/env python3
"""
MFLEs: Multi-Frequency Learning Ensemble
Ensemble forecasting using multiple frequency components.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot
from utils.ts_utils import load_ts_data, split_ts, create_time_features, create_lags, create_rolling_features, detect_frequency

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_frequency_components(data, config):
    """Extract features from multiple frequency components."""
    features = create_time_features(data)
    features = create_lags(data, config['model'].get('lags', [1, 2, 3, 7, 14]))
    features = create_rolling_features(
        data,
        windows=config['model'].get('windows', [7, 14, 30]),
        functions=config['model'].get('rolling_funcs', ['mean', 'std'])
    )
    
    return features


def create_ensemble_model(config):
    """Create ensemble model."""
    return RandomForestRegressor(
        n_estimators=config['model'].get('n_estimators', 100),
        max_depth=config['model'].get('max_depth', 10),
        random_state=config['model'].get('random_state', 42)
    )


def fit_and_predict(model, X_train, y_train, X_test, config):
    """Fit model and generate predictions."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def create_visualizations(data, train_data, test_data, predictions, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = setup_figure(config)
    
    ax.plot(train_data.index, train_data.values,
            c=config['plotting']['style']['colors']['primary'],
            linewidth=config['plotting']['style']['linewidth'],
            alpha=config['plotting']['style']['alpha'],
            label='Train')
    
    [ax.plot(test_data.index, test_data.values,
             c=config['plotting']['style']['colors']['accent'],
             linewidth=config['plotting']['style']['linewidth'],
             alpha=config['plotting']['style']['alpha'],
             label='Test')
     for _ in [None] if test_data is not None]
    
    forecast_index = pd.date_range(
        start=test_data.index[0] if test_data is not None else train_data.index[-1] + pd.Timedelta(days=1),
        periods=len(predictions),
        freq=detect_frequency(train_data)
    )
    
    ax.plot(forecast_index, predictions,
            c=config['plotting']['style']['colors']['secondary'],
            linewidth=config['plotting']['style']['linewidth'],
            label='Forecast')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'mfles_forecast.png', config)
     for _ in [None] if config['output']['save_plots']]
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    data = load_ts_data(
        Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    
    features = extract_frequency_components(data, config)
    features = features.dropna()
    
    X = features.drop('value', axis=1)
    y = features['value']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - config['model'].get('train_size', 0.8),
        random_state=config['model'].get('random_state', 42)
    )
    
    model = create_ensemble_model(config)
    predictions = fit_and_predict(model, X_train, y_train, X_test, config)
    
    train_data = data.iloc[:len(y_train)]
    test_data = data.iloc[len(y_train):len(y_train) + len(y_test)] if len(y_test) > 0 else None
    
    create_visualizations(data, train_data, test_data, predictions, config)
    
    print("✓ MFLEs ensemble forecasting complete")


if __name__ == "__main__":
    main()

