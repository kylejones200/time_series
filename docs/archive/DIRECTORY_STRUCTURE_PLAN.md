# Time Series Project Directory Structure Plan

## Proposed Organization

### 1. **forecasting/**
- Classical forecasting methods (ARIMA, ARAR, exponential smoothing)
- Modern forecasting (NBEATS, Chronos, TimesFM, MOMENT)
- Ensemble methods and model comparisons
- Files: ARIMA/, forecasting.ipynb, Amazon Chronos model.ipynb, etc.

### 2. **anomaly_detection/**
- Anomaly detection methods and tools
- Merlion, PyOD, STUMPY implementations
- Files: Merlion/, anomaly detection notebooks

### 3. **financial_time_series/**
- Stock market, commodity, and economic data analysis
- Gold, oil, treasury bonds, inflation analysis
- Files: Gold pricing.ipynb, Shell and Brent Crude regression.ipynb, etc.

### 4. **energy_sector/**
- ERCOT data analysis
- Oil production (Bakken, North Dakota)
- Energy forecasting and decline curve analysis
- Files: ERCOT notebooks, oil production/, Bakken analysis

### 5. **digital_humanities/**
- Sentiment analysis of historical texts
- Democracy, war sentiment, linguistic change
- Files: Digital Humanities Sentiment/, war sentiment.ipynb, etc.

### 6. **neural_networks/**
- Deep learning for time series
- LSTM, RNN, KAN implementations
- Files: neural networks in python.ipynb, kan/, torch_geometric_temporal.ipynb

### 7. **econometrics/**
- Causal inference, regression discontinuity
- Panel data, VAR models, Granger causality
- Files: regression discontinuity.ipynb, Granger.ipynb, etc.

### 8. **visualization/**
- Animation and plotting utilities
- Interactive visualizations
- Files: animation plots.ipynb, matplotlib gradient.ipynb, visualization.py

### 9. **utilities/**
- Helper functions and shared code
- Data preprocessing utilities
- Files: project_utils.py, timeseries_utils.py, preprocessing.py

### 10. **experiments/**
- Experimental and exploratory work
- One-off analyses and prototypes
- Files: dev/, experimental notebooks

## Standard Structure for Each Project Directory:
```
project_name/
├── README.md              # Project description and overview
├── notebooks/             # Jupyter notebooks
├── data/                  # Data files (.csv, .xlsx, .numbers)
├── images/               # Generated plots and visualizations
├── code/                 # Python scripts and utilities
└── results/              # Output files and model artifacts
```

## Implementation Steps:
1. Create main project directories
2. Move files to appropriate directories
3. Create README.md for each project area
4. Organize data files into data/ subdirectories
5. Move images to images/ subdirectories
6. Clean up and consolidate similar notebooks
