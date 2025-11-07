#!/usr/bin/env python3
"""
Volatility Models (ARCH/GARCH)
Volatility forecasting using ARCH and GARCH models for financial time series.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index, split_ts

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_volatility_model(data, config):
    """Create ARCH/GARCH volatility model."""
    model_type = config['model']['type']
    
    model_map = {
        'ARCH': lambda: arch_model(
            data,
            vol=model_type,
            p=config['model']['p'],
            q=0,
            dist=config['model']['distribution']
        ),
        'GARCH': lambda: arch_model(
            data,
            vol=model_type,
            p=config['model']['p'],
            q=config['model']['q'],
            dist=config['model']['distribution']
        ),
        'EGARCH': lambda: arch_model(
            data,
            vol='EGARCH',
            p=config['model']['p'],
            q=config['model']['q'],
            dist=config['model']['distribution']
        ),
    }
    
    return model_map.get(model_type, model_map['GARCH'])()


def fit_and_forecast(model, config):
    """Fit model and generate volatility forecasts."""
    fitted_model = model.fit(
        update_freq=config['model'].get('update_freq', 1),
        disp=config['model'].get('disp', 'off')
    )
    
    forecast = fitted_model.forecast(horizon=config['model']['forecast_horizon'])
    forecast_variance = forecast.variance.iloc[-1].values
    forecast_volatility = np.sqrt(forecast_variance)
    
    return fitted_model, forecast_variance, forecast_volatility


def create_visualizations(data, train_data, test_data, fitted_model, forecast_variance, forecast_volatility, config):
    """Generate visualizations for volatility model."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for ax in axes.flatten():
        apply_plot_style(ax, {'plotting': config['plotting']})
    
    axes[0, 0].plot(train_data.index, train_data.values,
                    'k-', linewidth=config['plotting']['linewidth'],
                    alpha=config['plotting']['alpha'], label='Train Returns')
    if test_data is not None:
        axes[0, 0].plot(test_data.index, test_data.values,
                        'g-', linewidth=config['plotting']['linewidth'],
                        alpha=config['plotting']['alpha'], label='Test Returns')
    axes[0, 0].set_title('Time Series Returns')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Returns')
    apply_legend(axes[0, 0], config['plotting']['legend'])
    
    conditional_vol = fitted_model.conditional_volatility
    axes[0, 1].plot(train_data.index, conditional_vol,
                    'r-', linewidth=config['plotting']['linewidth'],
                    alpha=config['plotting']['alpha'], label='Conditional Volatility')
    axes[0, 1].set_title('Conditional Volatility (Fitted)')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Volatility')
    apply_legend(axes[0, 1], config['plotting']['legend'])
    
    forecast_index = pd.date_range(
        start=train_data.index[-1] + pd.Timedelta(days=1),
        periods=len(forecast_variance),
        freq='D'
    )
    
    axes[1, 0].plot(forecast_index, forecast_variance,
                    'b-', linewidth=config['plotting']['linewidth'],
                    marker='o', markersize=config['plotting']['markersize'],
                    label='Forecasted Variance')
    axes[1, 0].set_title('Forecasted Variance')
    axes[1, 0].set_xlabel('Forecast Horizon')
    axes[1, 0].set_ylabel('Variance')
    apply_legend(axes[1, 0], config['plotting']['legend'])
    
    axes[1, 1].plot(forecast_index, forecast_volatility,
                    'm-', linewidth=config['plotting']['linewidth'],
                    marker='o', markersize=config['plotting']['markersize'],
                    label='Forecasted Volatility')
    axes[1, 1].set_title('Forecasted Volatility (sqrt of variance)')
    axes[1, 1].set_xlabel('Forecast Horizon')
    axes[1, 1].set_ylabel('Volatility')
    apply_legend(axes[1, 1], config['plotting']['legend'])
    
    plt.tight_layout()
    
    output_path = output_dir / "volatility_forecast.png"
    save_plot(fig, output_path)
    plt.show()


def main():
    """Main execution function."""
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col=config['data']['date_col'])
    
    if config['data']['compute_returns']:
        returns = df[config['data']['value_col']].pct_change().dropna()
    else:
        returns = df[config['data']['value_col']]
    
    train_returns, test_returns = split_ts(returns, test_size=config['data']['test_size'])
    
    model = create_volatility_model(train_returns.values, config)
    
    print(f"\nFitting {config['model']['type']} model...")
    fitted_model, forecast_variance, forecast_volatility = fit_and_forecast(model, config)
    
    print("\nVolatility Model Results:")
    print("=" * 70)
    print(fitted_model.summary())
    
    print(f"\nForecast Statistics:")
    print(f"Mean Forecasted Variance: {np.mean(forecast_variance):.6f}")
    print(f"Mean Forecasted Volatility: {np.mean(forecast_volatility):.6f}")
    print(f"Forecast Horizon: {config['model']['forecast_horizon']} steps")
    
    create_visualizations(
        returns, train_returns, test_returns,
        fitted_model, forecast_variance, forecast_volatility, config
    )
    
    print(f"✓ {config['model']['type']} volatility forecasting complete")


if __name__ == "__main__":
    main()

