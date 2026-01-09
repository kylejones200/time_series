#!/usr/bin/env python3
"""Plotting utilities for time series forecasts."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def create_forecast_plot(
    train: pd.Series,
    test: Optional[pd.Series] = None,
    forecast: Optional[pd.Series] = None,
    conf_int: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Forecast",
    xlabel: str = "Date",
    ylabel: str = "Value",
    train_label: str = "Historical (Train)",
    test_label: str = "Actual (Test)",
    forecast_label: str = "Forecast",
    show_ci: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create standardized forecast plot.
    
    This consolidates plotting code that was duplicated across templates.
    
    Parameters:
    -----------
    train : pd.Series
        Training/historical data
    test : pd.Series, optional
        Test/actual data to compare against
    forecast : pd.Series, optional
        Forecasted values
    conf_int : pd.DataFrame, optional
        Confidence intervals with 'lower' and 'upper' columns
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    train_label : str
        Label for training data
    test_label : str
        Label for test data
    forecast_label : str
        Label for forecast
    show_ci : bool
        Whether to show confidence intervals
        
    Returns:
    --------
    tuple
        (figure, axes) objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot historical data
    ax.plot(
        train.index,
        train.values,
        "k-",
        linewidth=1.5,
        label=train_label,
        alpha=0.8,
    )
    
    # Plot test data if provided
    if test is not None:
        ax.plot(
            test.index,
            test.values,
            "g-",
            linewidth=1.5,
            label=test_label,
            alpha=0.8,
        )
    
    # Plot forecast if provided
    if forecast is not None:
        ax.plot(
            forecast.index,
            forecast.values,
            "r--",
            linewidth=1.5,
            label=forecast_label,
            alpha=0.8,
        )
        
        # Plot confidence intervals if provided
        if conf_int is not None and show_ci:
            ax.fill_between(
                conf_int.index,
                conf_int["lower"],
                conf_int["upper"],
                color="r",
                alpha=0.2,
                label="95% Confidence Interval",
            )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def save_plot(
    fig: plt.Figure,
    output_path: Path,
    dpi: int = 300,
    bbox_inches: str = "tight",
    facecolor: str = "white",
    close: bool = True,
) -> Path:
    """
    Save plot with standard settings.
    
    Consolidates the repeated fig.savefig() pattern.
    Properly handles file closing on Windows to avoid file locking issues.
    
    Parameters:
    -----------
    fig : matplotlib.Figure
        Figure to save
    output_path : Path
        Output file path
    dpi : int
        DPI for saved figure
    bbox_inches : str
        Bbox inches setting
    facecolor : str
        Face color for saved figure
    close : bool
        Whether to close the figure after saving (default: True).
        Set to False if you want to display the figure with plt.show() afterward.
        However, closing is recommended on Windows to avoid file locking issues.
        
    Returns:
    --------
    Path
        Output file path
    """
    from pathlib import Path as PathType
    
    output_path = PathType(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor=facecolor,
    )
    
    # Flush to ensure data is written before closing (important on Windows)
    if fig.canvas is not None:
        fig.canvas.flush_events()
    
    # Close figure if requested to avoid file locking issues on Windows
    # Note: If you need to show the plot, set close=False or recreate it
    if close:
        plt.close(fig)
    
    return output_path

