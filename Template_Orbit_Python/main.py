#!/usr/bin/env python3
"""
Orbit: Bayesian Time Series Forecasting
Bayesian structural time series models for forecasting.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import util
from orbit.models import DLT, KTR, LGT
from orbit.diagnostics.plot import plot_predicted_data


def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj


plotting_utils = repo_import("utils.plotting_utils")
setup_figure = plotting_utils.setup_figure
apply_legend = plotting_utils.apply_legend
save_plot = plotting_utils.save_plot


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(config):
    """Load time series data."""
    data_path = Path(__file__).parent.parent / 'data' / Path(config['data']['input_file']).name
    df = pd.read_csv(data_path)
    
    date_col = config['data']['date_col']
    value_col = config['data']['value_col']
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: 'date', value_col: 'value'})
    
    return df


def create_model(config):
    """Create Orbit model based on config."""
    model_map = {
        'DLT': DLT,
        'KTR': KTR,
        'LGT': LGT,
    }
    
    model_class = model_map[config['model']['type']]
    model_params = {
        'response_col': 'value',
        'date_col': 'date',
        **config['model'].get('params', {})
    }
    
    return model_class(**model_params)


def fit_and_predict(model, df, config):
    """Fit model and generate predictions."""
    model.fit(train_df=df)
    predictions = model.predict(df=df)
    return predictions


def create_visualizations(df, predictions, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = setup_figure(config)
    
    ax.plot(df['date'], df['value'], 
            c=config['plotting']['style']['colors']['primary'],
            linewidth=config['plotting']['style']['linewidth'],
            alpha=config['plotting']['style']['alpha'],
            label='Actual')
    
    ax.plot(predictions['date'], predictions['prediction'],
            c=config['plotting']['style']['colors']['secondary'],
            linewidth=config['plotting']['style']['linewidth'],
            label='Forecast')
    
    pred_cols = [col for col in predictions.columns if 'prediction_' in col and col.replace('prediction_', '').isdigit()]
    pred_lower = predictions.get('prediction_5') or predictions.get('prediction_lower')
    pred_upper = predictions.get('prediction_95') or predictions.get('prediction_upper')
    
    [ax.fill_between(predictions['date'], pred_lower, pred_upper,
                     alpha=0.2, color=config['plotting']['style']['colors']['secondary'])
     for _ in [None] if pred_lower is not None and pred_upper is not None]
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    [save_plot(fig, output_dir / 'orbit_forecast.png', config)
     for _ in [None] if config['output']['save_plots']]
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    df = load_data(config)
    model = create_model(config)
    predictions = fit_and_predict(model, df, config)
    create_visualizations(df, predictions, config)
    
    print("✓ Orbit forecasting complete")


if __name__ == "__main__":
    main()

