# Time Series Templates

A comprehensive collection of time series forecasting and analysis templates, organized using a DRY (Don't Repeat Yourself) structure.

## 🎯 Overview

This repository contains **34 production-ready Python templates** for time series analysis, from classical statistical methods to cutting-edge deep learning approaches. Each template is self-contained, config-driven, and follows pythonic best practices.

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
10. **Nixtla** - StatsForecast fast statistical forecasting

### Modern Forecasting Libraries
11. **Prophet** - Facebook's Prophet
12. **Greykite** - LinkedIn's forecasting library
13. **Darts** - Unified forecasting interface
14. **PyCaret** - Low-code time series forecasting

### Deep Learning
15. **LSTM** - Long Short-Term Memory networks for time series
16. **NBEATS** - Neural Basis Expansion Analysis
17. **TSAI** - Deep learning for time series
18. **BERT** - Time series classification with BERT

### Feature Engineering & Analysis
19. **TSFresh** - Automated feature extraction
20. **Aeon** - Time series analysis toolkit
21. **Kalman** - State space models (Kalman filters)

### Specialized
22. **Merlion** - Forecasting & anomaly detection (enhanced with ARIMA and AutoEncoder)
23. **MFLEs** - Multi-frequency learning ensemble
24. **Autogluon** - Automated time series forecasting
25. **Econometrics** - Causal inference and econometric methods (Granger, RDD, OLS, VAR)
26. **CCM** - Convergent Cross Mapping for causal inference
27. **SparseRegression** - LASSO/Ridge/Elastic Net with automatic feature selection
28. **STUMPY_PyOD** - Matrix profile and outlier detection
29. **BollingerBands** - Technical indicators
30. **SerialCorrelation** - Serial correlation tests and corrections
31. **ConfidenceIntervals** - Bootstrap and parametric confidence intervals
32. **RegimeSwitching** - Markov switching models
33. **TimeSeriesDecomposition** - Trend, seasonal, and residual decomposition
34. **tslearn** - Time series machine learning (clustering, DTW)

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
6. **Pythonic**: Clean, idiomatic Python code (dictionary dispatch, list comprehensions)
7. **No Data Leakage**: Proper time series splitting and preprocessing

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
