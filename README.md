# Time Series Templates

A comprehensive collection of time series forecasting and analysis templates, organized using a DRY (Don't Repeat Yourself) structure.

## 🎯 Overview

This repository contains **48 production-ready Python templates** for time series analysis, from classical statistical methods to cutting-edge deep learning approaches. Each template is self-contained, config-driven, and follows pythonic best practices.

## 📁 Structure

```
time_series/
├── utils/                    # Shared utilities (DRY)
│   ├── ts_utils.py          # Time series utilities (date handling, feature engineering)
│   └── plotting_utils.py    # Plotting functions
├── data/                     # Shared data directory
├── Template_*_Python/        # Individual concept folders
│   ├── main.py               # Main execution script
│   ├── config.yaml           # Configuration file
│   ├── requirements.txt      # Project-specific dependencies
│   ├── README.md             # Project documentation
│   └── outputs/              # Output directory
└── WIP/                      # Work in progress (legacy files)
```

## 🚀 Quick Start

1. **Choose a template** from the list below
2. **Navigate to the template directory**:
   ```bash
   cd Template_Prophet_Python
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Place your data** in the shared `data/` directory
5. **Update `config.yaml`** with your data file name and column names
6. **Run**:
   ```bash
   python main.py
   ```

## 📚 Available Templates

### Classical Methods
1. **ARIMA** - Autoregressive Integrated Moving Average
2. **ARAR** - Autoregressive Autoregressive (reduced lag sets)
3. **MovingAverage** - Simple and exponential moving averages
4. **ExponentialSmoothing** - Simple, Double, and Triple (Holt-Winters) exponential smoothing
5. **BoxJenkins** - Systematic Box-Jenkins methodology for ARIMA
6. **VAR** - Vector Autoregression for multivariate time series

### Bayesian & Statistical
7. **Bayesian** - PyMC Bayesian time series modeling
8. **BayesianChangePoint** - Bayesian change point detection (MCMC)
9. **Orbit** - Bayesian structural time series
10. **PyBSTS** - Bayesian structural time series (pybsts, alternative to Orbit)
11. **Nixtla** - StatsForecast fast statistical forecasting

### Modern Forecasting Libraries
12. **Prophet** - Facebook's Prophet
13. **Greykite** - LinkedIn's forecasting library
14. **Darts** - Unified forecasting interface
15. **PyCaret** - Low-code time series forecasting
16. **Sundial** - Transformer-based forecasting (THUML Sundial)
17. **Chronos** - Amazon Chronos transformer forecasting
18. **TimesFM** - Google TimesFM foundation model
19. **LagLlama** - Lag-Llama foundation model forecasting
20. **StatsForecast** - Nixtla statsforecast AutoARIMA

### Deep Learning
21. **LSTM** - Long Short-Term Memory networks for time series
22. **NBEATS** - Neural Basis Expansion Analysis
23. **TSAI** - Deep learning for time series
24. **BERT** - Time series classification with BERT

### Feature Engineering & Analysis
25. **TSFresh** - Automated feature extraction
26. **Aeon** - Time series analysis toolkit
27. **Kalman** - State space models (Kalman filters)
28. **Differencing** - Differencing diagnostics and ADF tests
29. **IrregularSeries** - Resampling and Gaussian Process interpolation for irregular data
30. **ForecastErrorAnalysis** - ETS-based forecast error diagnostics

### Specialized
31. **Merlion** - Forecasting & anomaly detection (enhanced with ARIMA and AutoEncoder)
32. **MFLEs** - Multi-frequency learning ensemble
33. **Autogluon** - Automated time series forecasting
34. **Econometrics** - Causal inference and econometric methods (Granger, RDD, OLS, VAR)
35. **CCM** - Convergent Cross Mapping for causal inference
36. **SparseRegression** - LASSO/Ridge/Elastic Net with automatic feature selection
37. **STUMPY_PyOD** - Matrix profile and outlier detection
38. **BollingerBands** - Technical indicators
39. **SerialCorrelation** - Serial correlation tests and corrections
40. **ConfidenceIntervals** - Bootstrap and parametric confidence intervals
41. **RegimeSwitching** - Markov switching models
42. **TimeSeriesDecomposition** - Trend, seasonal, and residual decomposition
43. **tslearn** - Time series machine learning (clustering, DTW)
44. **Volatility** - ARCH/GARCH volatility modeling
45. **TransferEntropy** - Information-theoretic causal inference
46. **Copula** - Multivariate dependency modeling with copulas
47. **PyTimeTK** - Feature engineering toolkit for time series
48. **OrderedEvaluation** - Scoring ordinal forecasts and policy impact

## 🛠️ Shared Utilities

### `utils/ts_utils.py`
Common time series operations:
- `load_ts_data()` - Load time series from CSV
- `ensure_datetime_index()` - Ensure datetime index
- `resample_ts()` - Resample to different frequencies
- `create_lags()` - Create lagged features
- `create_rolling_features()` - Rolling window statistics
- `create_time_features()` - Time-based features (year, month, day, etc.)
- `split_ts()` - Time series train/test split
- `detect_frequency()` - Detect time series frequency
- `fill_missing_dates()` - Fill missing dates
- `remove_outliers_iqr()` - Outlier removal

### `utils/plotting_utils.py`
Consistent visualization:
- `setup_figure()` - Create styled figure
- `apply_plot_style()` - Apply matplotlib styling
- `apply_legend()` - Configure legend
- `save_plot()` - Save plots with config

## ⚙️ Configuration

Each template has a `config.yaml` with:
- **Data**: Input file path and column names
- **Model**: Model-specific parameters
- **Plotting**: Matplotlib styling (spines, grid, colors, etc.)
- **Output**: Plot settings

All templates use config-driven styling for consistent, minimalist visualizations.

## ✨ Features

1. **DRY Structure**: Shared utilities eliminate code duplication
2. **Consistency**: All templates follow the same patterns
3. **Maintainability**: Update utilities once, all templates benefit
4. **Self-Contained**: Each template has its own `requirements.txt`
5. **Config-Driven**: Easy customization via YAML


## 📊 Data Format

All templates expect CSV files with:
- A date column (configurable name)
- One or more value columns (configurable names)
- Data sorted by date

Place your data files in the shared `data/` directory.

## 🔧 Requirements

Each template has its own `requirements.txt`. Common dependencies include:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `scikit-learn` - Machine learning utilities
- `pyyaml` - Configuration files

## 📝 Notes

- All templates import from shared `utils/` directory
- All templates load data from shared `data/` directory
- Outputs are saved to each template's `outputs/` directory
- Templates are designed to be production-ready and well-documented

## 🤝 Contributing

This is a personal research collection, but feel free to:
- Explore the methodologies
- Adapt code for your projects
- Suggest improvements or report issues

## 📄 License

MIT License - See individual files for specific attributions and data source licenses.
