#!/usr/bin/env python3
"""
TSAI: Time Series AI
Deep learning for time series using TSAI library.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from tsai.all import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot
from utils.ts_utils import load_ts_data, split_ts, detect_frequency

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(data, config):
    """Prepare data for TSAI."""
    train_size = config['model'].get('train_size', 0.8)
    train, test = split_ts(data, train_size=train_size)
    
    X_train = train.values.reshape(-1, 1, len(train))
    X_test = test.values.reshape(-1, 1, len(test)) if test is not None else None
    
    y_train = train.values
    y_test = test.values if test is not None else None
    
    return X_train, X_test, y_train, y_test


def create_model(X_train, config):
    """Create TSAI model."""
    model_type = config['model'].get('type', 'InceptionTime')
    
    model_map = {
        'InceptionTime': InceptionTime,
        'ResNet': ResNet,
        'XceptionTime': XceptionTime,
        'ROCKET': ROCKET,
    }
    
    model_class = model_map.get(model_type, InceptionTime)
    
    return model_class(c_in=1, c_out=1)


def train_model(model, X_train, y_train, config):
    """Train TSAI model."""
    learn = Learner(
        TSDataLoaders.from_arrays(X_train, y_train, bs=config['model'].get('batch_size', 64)),
        model,
        metrics=[mae, rmse]
    )
    
    learn.fit_one_cycle(
        n_epoch=config['model'].get('n_epochs', 10),
        lr_max=config['model'].get('lr', 1e-3)
    )
    
    return learn


def create_visualizations(data, train_data, predictions, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = setup_figure(config)
    
    ax.plot(train_data.index, train_data.values,
            c=config['plotting']['style']['colors']['primary'],
            linewidth=config['plotting']['style']['linewidth'],
            alpha=config['plotting']['style']['alpha'],
            label='Historical')
    
    forecast_index = pd.date_range(
        start=train_data.index[-1] + pd.Timedelta(days=1),
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
    
    [save_plot(fig, output_dir / 'tsai_forecast.png', config)
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
    
    X_train, X_test, y_train, y_test = prepare_data(data, config)
    model = create_model(X_train, config)
    learn = train_model(model, X_train, y_train, config)
    
    predictions = learn.get_preds()[0].numpy().flatten()
    create_visualizations(data, data.iloc[:len(y_train)], predictions, config)
    
    print("✓ TSAI deep learning complete")


if __name__ == "__main__":
    main()

