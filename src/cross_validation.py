#!/usr/bin/env python3
"""Time series cross-validation utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TimeSeriesCrossValidator:
    """
    Time-aware cross-validation for time series.
    
    Ensures proper temporal ordering and prevents data leakage.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        """
        Initialize time series cross-validator.
        
        Parameters:
        -----------
        n_splits : int
            Number of splits
        test_size : int, optional
            Size of test set in each split
        gap : int
            Gap between train and test sets (default: 0)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    
    def split(self, X: pd.Series | pd.DataFrame | np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Parameters:
        -----------
        X : pd.Series, pd.DataFrame, or np.ndarray
            Time series data
        
        Returns:
        --------
        List[Tuple]
            List of (train_indices, test_indices) tuples
        """
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
        
        return list(self.tscv.split(X))
    
    def evaluate(
        self,
        X: pd.Series | pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        model_factory: Callable,
        fit_func: Callable,
        predict_func: Callable,
    ) -> pd.DataFrame:
        """
        Perform cross-validation and return metrics.
        
        Parameters:
        -----------
        X : pd.Series, pd.DataFrame, or np.ndarray
            Features
        y : pd.Series or np.ndarray
            Target values
        model_factory : Callable
            Function that returns a new model instance
        fit_func : Callable
            Function(model, X_train, y_train) -> fitted_model
        predict_func : Callable
            Function(model, X_test) -> predictions
        
        Returns:
        --------
        pd.DataFrame
            Metrics for each fold
        """
        splits = self.split(X)
        results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx], X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
            y_train, y_test = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx], y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
            
            # Train model
            model = model_factory()
            fitted_model = fit_func(model, X_train, y_train)
            
            # Predict
            y_pred = predict_func(fitted_model, X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "MSE": mse,
                "MAE": mae,
                "RMSE": rmse,
                "R²": r2,
            })
        
        return pd.DataFrame(results)
    
    def plot_cv_splits(
        self,
        series: pd.Series,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Visualize cross-validation splits.
        
        Parameters:
        -----------
        series : pd.Series
            Time series to visualize
        figsize : tuple
            Figure size
        
        Returns:
        --------
        plt.Figure
            Figure with CV splits visualization
        """
        splits = self.split(series)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot full series
        ax.plot(series.index, series.values, "k-", alpha=0.3, label="Full Series")
        
        # Color-code train/test splits
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_splits))
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_series = series.iloc[train_idx]
            test_series = series.iloc[test_idx]
            
            ax.plot(
                train_series.index,
                train_series.values,
                color=colors[fold_idx],
                alpha=0.5,
                linewidth=2,
                label=f"Fold {fold_idx+1} Train",
            )
            ax.plot(
                test_series.index,
                test_series.values,
                color=colors[fold_idx],
                alpha=0.8,
                linewidth=2,
                linestyle="--",
                label=f"Fold {fold_idx+1} Test",
            )
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title(f"Time Series Cross-Validation Splits (n_splits={self.n_splits})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

