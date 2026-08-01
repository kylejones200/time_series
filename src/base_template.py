#!/usr/bin/env python3
"""Base template class for all forecasting templates."""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt

from .config import load_config, get_output_dir
from .utils import ensure_output_dir
from .loader import load_time_series
from .plotting import create_forecast_plot, save_plot


class BaseTemplate:
    """
    Base class for all forecasting templates.
    
    This consolidates common functionality that was duplicated across templates:
    - Config loading
    - Data loading
    - Output directory management
    - Plot creation and saving
    - CSV saving
    """
    
    def __init__(self, config_path: str = "config.yaml", script_dir: Optional[Path] = None):
        """
        Initialize template with configuration.
        
        Parameters:
        -----------
        config_path : str
            Path to config.yaml file
        script_dir : Path, optional
            Directory where template script is located.
            If None, uses config_path parent directory.
        """
        self.config_path = Path(config_path)
        self.script_dir = script_dir or self.config_path.parent
        self.config = load_config(config_path)
        self.output_dir = ensure_output_dir(
            get_output_dir(self.config, self.script_dir)
        )
        
    def load_data(self) -> pd.Series:
        """
        Load time series data using standard loader.
        
        Returns:
        --------
        pd.Series
            Time series with datetime index
        """
        data_config = self.config.get("data", {})
        return load_time_series(
            data_config["input_file"],
            date_column=data_config.get("date_column", "date"),
            value_column=data_config.get("value_column", "value")
        )
    
    def create_plot(
        self,
        train: pd.Series,
        test: Optional[pd.Series] = None,
        forecast: Optional[pd.Series] = None,
        conf_int: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        **plot_kwargs
    ) -> tuple:
        """
        Create standardized forecast plot.
        
        Parameters:
        -----------
        train : pd.Series
            Training data
        test : pd.Series, optional
            Test data
        forecast : pd.Series, optional
            Forecast data
        conf_int : pd.DataFrame, optional
            Confidence intervals
        title : str, optional
            Plot title (defaults to config value)
        **plot_kwargs
            Additional arguments passed to create_forecast_plot
            
        Returns:
        --------
        tuple
            (figure, axes) objects
        """
        plotting_config = self.config.get("plotting", {})
        figure_size = plotting_config.get("figure_size", [12, 6])
        
        if title is None:
            title = f"Forecast Plot"
        
        return create_forecast_plot(
            train=train,
            test=test,
            forecast=forecast,
            conf_int=conf_int,
            figsize=tuple(figure_size),
            title=title,
            **plot_kwargs
        )
    
    def save_plot(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: Optional[int] = None,
    ) -> Path:
        """
        Save plot to output directory.
        
        Parameters:
        -----------
        fig : matplotlib.Figure
            Figure to save
        filename : str
            Output filename
        dpi : int, optional
            DPI (defaults to config value)
            
        Returns:
        --------
        Path
            Path to saved file
        """
        if dpi is None:
            dpi = self.config.get("output", {}).get("dpi", 300)
        
        output_path = self.output_dir / filename
        return save_plot(fig, output_path, dpi=dpi)
    
    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save DataFrame to CSV in output directory.
        
        Properly handles file closing on Windows to avoid file locking issues.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save
        filename : str
            Output filename
            
        Returns:
        --------
        Path
            Path to saved file
        """
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use explicit encoding and ensure file handle is properly closed
        # This is especially important on Windows
        df.to_csv(output_path, index=False, encoding="utf-8")
        
        return output_path

