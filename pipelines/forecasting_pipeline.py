#!/usr/bin/env python3
"""
Forecasting pipeline for production time series.

Unified API to run multiple forecasting models and compare results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml

from .model_registry import ModelRegistry, get_default_registry
from evaluation import ModelComparison, ModelResult, calculate_metrics


class ForecastingPipeline:
    """
    Unified forecasting pipeline.
    
    Loads production data, runs multiple forecasting models, and compares results.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        well_id: Optional[str] = None,
        target_column: str = "production_rate",
        date_column: str = "date",
        forecast_horizon: int = 12,
        train_size: Optional[Union[float, int]] = None,
        test_size: Optional[Union[float, int]] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize forecasting pipeline.
        
        Parameters:
        -----------
        data_path : str or Path
            Path to production data CSV
        well_id : str, optional
            Well ID to filter data (if multiple wells in file)
        target_column : str
            Column name for production rate
        date_column : str
            Column name for dates
        forecast_horizon : int
            Number of periods to forecast
        train_size : float or int, optional
            Training data size (float=proportion, int=number of periods)
        test_size : float or int, optional
            Test data size (if None, uses remaining data)
        registry : ModelRegistry, optional
            Model registry to use (defaults to global registry)
        """
        self.data_path = Path(data_path)
        self.well_id = well_id
        self.target_column = target_column
        self.date_column = date_column
        self.forecast_horizon = forecast_horizon
        self.train_size = train_size
        self.test_size = test_size
        self.registry = registry or get_default_registry()

        self.data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.Series] = None
        self.test_data: Optional[pd.Series] = None
        self.models: Dict[str, Any] = {}

    def load_data(self) -> "ForecastingPipeline":
        """
        Load and prepare production data.
        
        Returns:
        --------
        self
        """
        # Load CSV
        df = pd.read_csv(self.data_path)
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.set_index(self.date_column)

        # Filter by well_id if specified
        if self.well_id is not None and "well_id" in df.columns:
            df = df[df["well_id"] == self.well_id].copy()

        # Extract target column
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found. Available: {df.columns.tolist()}")

        production = df[self.target_column].sort_index()

        # Split train/test
        total_len = len(production)
        if self.test_size is not None:
            if isinstance(self.test_size, float):
                test_len = int(total_len * self.test_size)
            else:
                test_len = self.test_size
            train_len = total_len - test_len
        elif self.train_size is not None:
            if isinstance(self.train_size, float):
                train_len = int(total_len * self.train_size)
            else:
                train_len = self.train_size
            test_len = total_len - train_len
        else:
            # Default: use 80% for training
            train_len = int(total_len * 0.8)
            test_len = total_len - train_len

        self.train_data = production.iloc[:train_len]
        self.test_data = production.iloc[train_len:] if test_len > 0 else None
        self.data = production

        return self

    def add_model(self, name: str, model: Any):
        """
        Add a model to the pipeline.
        
        Parameters:
        -----------
        name : str
            Model name
        model : object
            Model instance with fit() and predict() methods
        """
        self.models[name] = model
        return self

    def add_model_from_registry(self, name: str):
        """
        Add a model from the registry.
        
        Parameters:
        -----------
        name : str
            Model name in registry
        """
        model = self.registry.get(name)
        self.add_model(name, model)
        return self

    def run_all(self) -> Dict[str, ModelResult]:
        """
        Run all registered models and generate forecasts.
        
        Returns:
        --------
        dict
            Dictionary mapping model names to ModelResult objects
        """
        if self.train_data is None:
            self.load_data()

        if not self.models:
            raise ValueError("No models registered. Use add_model() or add_model_from_registry()")

        results = {}

        for name, model in self.models.items():
            try:
                # Fit model
                model.fit(self.train_data)

                # Generate forecast
                forecast_start = self.train_data.index[-1] + pd.Timedelta(days=30)  # Approximate monthly
                forecast = model.predict(forecast_start, self.forecast_horizon, freq="MS")

                # Calculate metrics if test data available
                metrics = None
                if self.test_data is not None and len(forecast) > 0:
                    # Align test data with forecast
                    forecast_aligned = forecast.reindex(self.test_data.index, method="nearest")
                    valid_idx = ~forecast_aligned.isna() & ~self.test_data.isna()
                    if valid_idx.sum() > 0:
                        metrics = calculate_metrics(
                            self.test_data[valid_idx],
                            forecast_aligned[valid_idx],
                            name,
                        )

                # Get model parameters if available
                model_params = getattr(model, "params", None)

                results[name] = ModelResult(
                    name=name,
                    forecast=forecast,
                    metrics=metrics,
                    model_params=model_params,
                )
            except Exception as e:
                print(f"Error running model '{name}': {e}")
                continue

        return results

    def compare_models(self, results: Optional[Dict[str, ModelResult]] = None) -> pd.DataFrame:
        """
        Compare model results.
        
        Parameters:
        -----------
        results : dict, optional
            Model results (if None, runs all models first)
            
        Returns:
        --------
        pd.DataFrame
            Comparison table with metrics
        """
        if results is None:
            results = self.run_all()

        if not results:
            return pd.DataFrame()

        # Use test data for comparison if available
        comparison_data = self.test_data if self.test_data is not None else self.train_data

        comparison = ModelComparison()
        for result in results.values():
            comparison.add_result(result)

        return comparison.compare_metrics(comparison_data)

    def save_results(
        self,
        results: Dict[str, ModelResult],
        output_dir: Union[str, Path],
        prefix: str = "forecast",
    ):
        """
        Save forecast results to CSV files.
        
        Parameters:
        -----------
        results : dict
            Model results dictionary
        output_dir : str or Path
            Output directory
        prefix : str
            Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual forecasts
        for name, result in results.items():
            forecast_df = result.forecast.to_frame("forecast")
            if result.confidence_intervals is not None:
                forecast_df = pd.concat([forecast_df, result.confidence_intervals], axis=1)

            filename = output_dir / f"{prefix}_{name.replace(' ', '_').lower()}.csv"
            forecast_df.to_csv(filename)

        # Save comparison metrics
        comparison = self.compare_models(results)
        comparison.to_csv(output_dir / f"{prefix}_comparison.csv")

        print(f"Results saved to {output_dir}")

