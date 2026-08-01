Overview
========

The Time Series Forecasting library provides a comprehensive toolkit for forecasting production data, with a focus on comparing modern time series methods against traditional decline curve analysis (DCA).

Key Features
------------

**48 Production-Ready Templates**
   Each template implements a specific forecasting method (ARIMA, Prophet, LSTM, etc.) with a consistent interface.

**Unified Pipeline**
   Compare multiple models side-by-side with standardized evaluation metrics.

**DCA Integration**
   Built-in decline curve analysis models for baseline comparisons.

**Standardized Evaluation**
   Consistent metrics (MAE, RMSE, MAPE, R²) across all models.

**Config-Driven**
   All templates use YAML configuration files for easy customization.

Architecture
------------

The library is organized into several key components:

**Core Utilities (``src/``)**
   - Data loading and preprocessing
   - Model wrappers
   - Evaluation utilities
   - Plotting functions

**Models (``models/``)**
   - DCA models (Arps variants)
   - Extensible model interface

**Pipelines (``pipelines/``)**
   - Unified forecasting pipeline
   - Model registry
   - Comparison utilities

**Templates (``*_Python/``)**
   - Individual forecasting method implementations
   - Each with its own config, main script, and README

**Evaluation (``evaluation/``)**
   - Standard metrics
   - Model comparison tools

Workflow
--------

The typical workflow:

1. **Prepare Data**: Format your time series data as CSV with date and value columns
2. **Choose Template**: Select a forecasting method template
3. **Configure**: Edit ``config.yaml`` to specify data paths and parameters
4. **Run**: Execute ``main.py`` to generate forecasts
5. **Evaluate**: Review metrics and plots in ``outputs/``
6. **Compare**: Use the pipeline to compare multiple models

For more details, see:

- :doc:`data_format` for data preparation
- :doc:`pipeline_usage` for using the unified pipeline
- :doc:`model_selection` for choosing the right method
- :doc:`evaluation` for understanding metrics

