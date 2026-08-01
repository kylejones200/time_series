"""Source modules for time series forecasting pipeline."""

import signalplot
from matplotlib.axes import Axes

# Apply SignalPlot's clean defaults once at module level
# This eliminates the need for signalplot.apply() in every template
signalplot.apply()

# Team preference: disable gridlines across generated figures.
# We patch Axes.grid so template-level `ax.grid(...)` calls become no-ops.
if not hasattr(Axes, "_organizer_original_grid"):
    Axes._organizer_original_grid = Axes.grid

    def _no_grid(self, *args, **kwargs):
        return None

    Axes.grid = _no_grid

from .loader import load_time_series
from .model import ARIMAModel
from .evaluator import Evaluator
from .config import load_config, get_output_dir, parse_common_config, CommonConfig
from .utils import repo_import, ensure_output_dir, bootstrap_repo
from .plotting import create_forecast_plot, save_plot
from .base_template import BaseTemplate
from .predictive_maintenance import (
    calculate_rul,
    create_rul_labels,
    add_rolling_statistics,
    calculate_degradation_rate,
    prepare_pm_features,
)
from .cross_validation import TimeSeriesCrossValidator
from .run_logger import start_auto_run_logger
from .feature_engineering import (
    extract_time_features,
    create_lag_features,
    create_rolling_features,
    create_differenced_features,
    create_seasonal_features,
    prepare_features,
)
from .confidence_intervals import (
    bootstrap_confidence_intervals,
    parametric_confidence_intervals,
    compare_ci_methods,
)

__all__ = [
    "load_time_series",
    "ARIMAModel",
    "Evaluator",
    "load_config",
    "get_output_dir",
    "parse_common_config",
    "CommonConfig",
    "repo_import",
    "ensure_output_dir",
    "bootstrap_repo",
    "create_forecast_plot",
    "save_plot",
    "BaseTemplate",
    "calculate_rul",
    "create_rul_labels",
    "add_rolling_statistics",
    "calculate_degradation_rate",
    "prepare_pm_features",
    "TimeSeriesCrossValidator",
    "extract_time_features",
    "create_lag_features",
    "create_rolling_features",
    "create_differenced_features",
    "create_seasonal_features",
    "prepare_features",
    "bootstrap_confidence_intervals",
    "parametric_confidence_intervals",
    "compare_ci_methods",
    "start_auto_run_logger",
]

# Start default run logging for scripts that import src.
start_auto_run_logger()

