"""
Shared utilities for time series templates.
"""

from .plotting_utils import setup_figure, apply_legend, save_plot, apply_plot_style
from .ts_utils import (
    load_ts_data,
    ensure_datetime_index,
    resample_ts,
    create_lags,
    create_rolling_features,
    split_ts,
    detect_frequency,
    fill_missing_dates,
    remove_outliers_iqr,
    create_time_features,
)

__all__ = [
    'setup_figure',
    'apply_legend',
    'save_plot',
    'apply_plot_style',
    'load_ts_data',
    'ensure_datetime_index',
    'resample_ts',
    'create_lags',
    'create_rolling_features',
    'split_ts',
    'detect_frequency',
    'fill_missing_dates',
    'remove_outliers_iqr',
    'create_time_features',
]
