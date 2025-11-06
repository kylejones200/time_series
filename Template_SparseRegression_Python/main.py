#!/usr/bin/env python3
"""
Sparse Regression (LASSO) for Time Series
LASSO regression with automatic feature selection for time series forecasting.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from utils.ts_utils import load_ts_data, ensure_datetime_index, split_ts, create_lags, create_rolling_features

warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_sparse_model(config):
    """Create sparse regression model."""
    model_type = config['model']['type']
    tscv = TimeSeriesSplit(n_splits=config['model']['cv_splits'])
    
    model_map = {
        'lasso': lambda: LassoCV(
            cv=tscv,
            max_iter=config['model'].get('max_iter', 10000),
            random_state=config['model'].get('random_state', None),
            alphas=config['model'].get('alphas', None)
        ),
        'ridge': lambda: RidgeCV(
            cv=tscv,
            alphas=config['model'].get('alphas', [0.1, 1.0, 10.0, 100.0])
        ),
        'elastic_net': lambda: ElasticNetCV(
            cv=tscv,
            max_iter=config['model'].get('max_iter', 10000),
            random_state=config['model'].get('random_state', None),
            l1_ratio=config['model'].get('l1_ratio', [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0])
        ),
    }
    
    model = model_map.get(model_type, model_map['lasso'])()
    return make_pipeline(StandardScaler(), model)


def create_features(data, config):
    """Create feature matrix with lags and rolling features."""
    features = data.copy()
    
    if config['features']['create_lags']:
        lags = config['features'].get('lags', [1, 2, 3, 7, 14])
        features = create_lags(features, lags)
    
    if config['features']['create_rolling']:
        windows = config['features'].get('rolling_windows', [7, 14, 30])
        functions = config['features'].get('rolling_funcs', ['mean', 'std'])
        features = create_rolling_features(features, windows, functions)
    
    return features.dropna()


def create_visualizations(data, train_data, test_data, predictions, model, X, config):
    """Generate visualizations for sparse regression."""
    output_dir = Path(__file__).parent / config['output']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for ax in axes.flatten():
        apply_plot_style(ax, {'plotting': config['plotting']})
    
    axes[0, 0].plot(train_data.index, train_data.values,
                    'k-', linewidth=config['plotting']['linewidth'],
                    alpha=config['plotting']['alpha'], label='Train')
    
    if test_data is not None:
        axes[0, 0].plot(test_data.index, test_data.values,
                        'g-', linewidth=config['plotting']['linewidth'],
                        alpha=config['plotting']['alpha'], label='Test')
        
        forecast_index = test_data.index
        axes[0, 0].plot(forecast_index, predictions,
                        'r--', linewidth=config['plotting']['linewidth'],
                        label='Forecast')
    else:
        forecast_index = train_data.index
        axes[0, 0].plot(forecast_index, predictions,
                        'r--', linewidth=config['plotting']['linewidth'],
                        label='Forecast')
    
    axes[0, 0].set_title(config['plot_titles']['forecast'])
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value')
    apply_legend(axes[0, 0], config['plotting']['legend'])
    
    regressor = model.named_steps[config['model']['type']]
    coef = pd.Series(regressor.coef_, index=X.columns)
    selected = coef[coef != 0].sort_values()
    
    if len(selected) > 0:
        axes[0, 1].barh(range(len(selected)), selected.values,
                       edgecolor='black', alpha=0.7)
        axes[0, 1].set_yticks(range(len(selected)))
        axes[0, 1].set_yticklabels(selected.index, fontsize=8)
        axes[0, 1].set_title(f'Selected Features (n={len(selected)})')
        axes[0, 1].set_xlabel('Coefficient Value')
        axes[0, 1].axvline(x=0, color='k', linestyle='--', linewidth=1)
    else:
        axes[0, 1].text(0.5, 0.5, 'No features selected', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Selected Features')
    
    if test_data is not None:
        axes[1, 0].scatter(test_data.values, predictions,
                          alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
        min_val = min(test_data.values.min(), predictions.min())
        max_val = max(test_data.values.max(), predictions.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val],
                        'r--', linewidth=config['plotting']['linewidth'])
        axes[1, 0].set_title('Actual vs Predicted (Test)')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
    
    if hasattr(regressor, 'alpha_'):
        axes[1, 1].text(0.5, 0.7, f'Optimal α: {regressor.alpha_:.4f}',
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if hasattr(regressor, 'l1_ratio_'):
        axes[1, 1].text(0.5, 0.5, f'Optimal L1 Ratio: {regressor.l1_ratio_:.4f}',
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[1, 1].text(0.5, 0.3, f'Features Selected: {len(selected)}/{len(coef)}',
                   ha='center', va='center', transform=axes[1, 1].transAxes,
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Model Summary')
    
    plt.tight_layout()
    
    output_path = output_dir / "sparse_regression_analysis.png"
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
    
    train_df, test_df = split_ts(df, test_size=config['data']['test_size'])
    
    features_train = create_features(train_df[config['data']['value_col']], config)
    features_test = create_features(test_df[config['data']['value_col']], config) if test_df is not None else None
    
    X_train = features_train.drop(config['data']['value_col'], axis=1)
    y_train = features_train[config['data']['value_col']]
    
    if features_test is not None:
        X_test = features_test.drop(config['data']['value_col'], axis=1)
        y_test = features_test[config['data']['value_col']]
    else:
        X_test = None
        y_test = None
    
    model = create_sparse_model(config)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test) if X_test is not None else None
    
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    print(f"\nSparse Regression ({config['model']['type'].upper()}) Results:")
    print("=" * 70)
    print(f"\nTraining Set:")
    print(f"MAE: {train_mae:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"R²: {train_r2:.4f}")
    
    if test_pred is not None:
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\nTest Set:")
        print(f"MAE: {test_mae:.4f}")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"R²: {test_r2:.4f}")
    
    regressor = model.named_steps[config['model']['type']]
    coef = pd.Series(regressor.coef_, index=X_train.columns)
    selected = coef[coef != 0]
    
    print(f"\nFeature Selection:")
    print(f"Total features: {len(coef)}")
    print(f"Selected features: {len(selected)}")
    print(f"Sparsity: {(1 - len(selected) / len(coef)) * 100:.1f}%")
    
    if hasattr(regressor, 'alpha_'):
        print(f"Optimal regularization parameter (α): {regressor.alpha_:.4f}")
    
    if len(selected) > 0:
        print(f"\nTop 10 Selected Features:")
        for feature, coef_val in selected.abs().sort_values(ascending=False).head(10).items():
            print(f"  {feature}: {coef[feature]:.4f}")
    
    create_visualizations(
        df, train_df[config['data']['value_col']],
        test_df[config['data']['value_col']] if test_df is not None else None,
        test_pred if test_pred is not None else train_pred,
        model, X_train, config
    )
    
    print(f"✓ Sparse regression ({config['model']['type']}) analysis complete")


if __name__ == "__main__":
    main()

