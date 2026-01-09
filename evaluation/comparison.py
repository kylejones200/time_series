#!/usr/bin/env python3
"""
Model comparison utilities.

Compares multiple forecasting models and generates comparison reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings

from .metrics import calculate_metrics, MetricResult


@dataclass
class ModelResult:
    """Container for a single model's forecast and metrics."""

    name: str
    forecast: pd.Series
    metrics: Optional[MetricResult] = None
    confidence_intervals: Optional[pd.DataFrame] = None
    model_params: Optional[Dict] = None


class ModelComparison:
    """
    Compare multiple forecasting models.
    
    Stores results from multiple models and provides comparison utilities.
    """

    def __init__(self):
        self.results: List[ModelResult] = []

    def add_result(self, result: ModelResult):
        """Add a model result to comparison."""
        self.results.append(result)

    def compare_metrics(
        self, actual: pd.Series
    ) -> pd.DataFrame:
        """
        Compare all models' metrics.
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values to compare against
            
        Returns:
        --------
        pd.DataFrame
            Comparison table with metrics for each model
        """
        comparison_data = []

        for result in self.results:
            # Calculate metrics if not already done
            if result.metrics is None:
                result.metrics = calculate_metrics(actual, result.forecast, result.name)

            metrics_dict = result.metrics.to_dict()
            metrics_dict["Model"] = result.name
            comparison_data.append(metrics_dict)

        df = pd.DataFrame(comparison_data)
        df = df.set_index("Model")

        # Sort by RMSE (best first)
        df = df.sort_values("RMSE")

        return df

    def get_best_model(self, metric: str = "RMSE") -> Optional[ModelResult]:
        """
        Get the best performing model based on a metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for ranking ('RMSE', 'MAE', 'MAPE', 'R²')
            
        Returns:
        --------
        ModelResult or None
            Best model result
        """
        if not self.results:
            return None

        # Filter results with metrics
        results_with_metrics = [r for r in self.results if r.metrics is not None]
        if not results_with_metrics:
            return None

        # Determine if metric is higher-is-better or lower-is-better
        if metric == "R²":
            # Higher is better
            best_result = max(results_with_metrics, key=lambda r: r.metrics.r2)
        else:
            # Lower is better (RMSE, MAE, MAPE, MSE)
            metric_attr_map = {
                "RMSE": "rmse",
                "MAE": "mae",
                "MAPE": "mape",
                "MSE": "mse",
            }
            attr_name = metric_attr_map.get(metric, "rmse")
            best_result = min(results_with_metrics, key=lambda r: getattr(r.metrics, attr_name, np.inf))

        return best_result


def compare_models(
    actual: pd.Series,
    forecasts: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Quick comparison function for multiple forecasts.
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    forecasts : dict
        Dictionary mapping model names to forecast Series
        
    Returns:
    --------
    pd.DataFrame
        Comparison table with metrics
    """
    comparison = ModelComparison()

    for name, forecast in forecasts.items():
        metrics = calculate_metrics(actual, forecast, name)
        result = ModelResult(name=name, forecast=forecast, metrics=metrics)
        comparison.add_result(result)

    return comparison.compare_metrics(actual)

