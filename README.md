# Time Series Forecasting for Production Data

A comprehensive forecasting pipeline that **compares time series methods against traditional decline curve analysis (DCA)** for oil & gas production forecasting.

## Overview

This repository combines:
- **48 production-ready Python templates** for time series forecasting (ARIMA, Prophet, deep learning, etc.)
- **Decline Curve Analysis (DCA) models** (Arps exponential, hyperbolic, harmonic)
- **Unified forecasting pipeline** to compare multiple models side-by-side
- **Standardized evaluation** with consistent metrics (MSE, MAE, MAPE, R²)

Use it to:
- Generate production forecasts using modern time series methods
- Compare time series forecasts against traditional DCA models
- Evaluate model performance with consistent metrics
- Select the best model for your production data

## Environment Setup

Use a single virtual environment (specialty venvs were consolidated):

```bash
cd time_series
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt   # optional extras (tensorflow, torch, etc.)
```

After `pip install -e .`, templates can `import src` without manual path hacks.

## Quick Start

### Option 1: Interactive Wizard (Recommended for New Users)

```bash
python scripts/quick_start.py
```

This interactive wizard will:
- Help you locate and validate your data
- Recommend suitable templates
- Guide you through your first forecast

### Option 2: Unified CLI

```bash
# List all available templates
python forecast.py list

# Validate your data
python forecast.py validate data/my_series.csv

# Get template recommendations based on your data
python forecast.py recommend data/my_series.csv

# Run a specific template
python forecast.py run ARIMA_Python --data data/my_series.csv

# Benchmark multiple templates
python forecast.py benchmark --data data/my_series.csv
```

### Option 3: Traditional Method

1. **Install Dependencies**

```bash
pip install pandas numpy scipy scikit-learn matplotlib signalplot pyyaml
```

2. **Generate Example Data**

```bash
python data/production/generate_example_data.py
```

3. **Run Comparison Example**

```bash
python examples/ts_vs_dca_comparison.py
```

### 4. Use the Pipeline in Your Code

```python
from pipelines import ForecastingPipeline, register_model
from models.dca import ArpsExponential, ArpsHyperbolic

# Register models
register_model("Arps Exponential", lambda: ArpsExponential())
register_model("Arps Hyperbolic", lambda: ArpsHyperbolic())

# Initialize pipeline
pipeline = ForecastingPipeline(
    data_path="data/production/well_production.csv",
    target_column="oil_rate",
    forecast_horizon=12,  # months
    train_size=0.8
)

# Add models
pipeline.add_model_from_registry("Arps Exponential")
pipeline.add_model_from_registry("Arps Hyperbolic")

# Run and compare
results = pipeline.run_all()
comparison = pipeline.compare_models(results)
print(comparison)

# Save results
pipeline.save_results(results, "outputs/well_001/")
```

## Repository Structure

```
time_series/
├── pipelines/                   # NEW: Unified forecasting pipeline
│   ├── forecasting_pipeline.py  # Main pipeline orchestrator
│   ├── model_registry.py        # Model registry system
│   └── __init__.py
│
├── models/                      # NEW: Model implementations
│   ├── dca/                     # Decline Curve Analysis models
│   │   ├── arps.py              # Arps models (exponential, hyperbolic, harmonic)
│   │   ├── exponential.py
│   │   └── hyperbolic.py
│   └── __init__.py
│
├── evaluation/                  # NEW: Evaluation and comparison
│   ├── metrics.py               # Standard metrics (MSE, MAE, MAPE, R²)
│   ├── comparison.py            # Model comparison utilities
│   └── __init__.py
│
├── examples/                    # NEW: Complete examples
│   └── ts_vs_dca_comparison.py  # Full comparison example
│
├── data/
│   ├── production/              # NEW: Production data examples
│   │   ├── well_production.csv  # Single well example
│   │   ├── multi_well.csv       # Multiple wells
│   │   ├── generate_example_data.py
│   │   └── README.md
│   └── ...                      # Other datasets
│
├── utils/                       # Shared utilities (DRY)
│   ├── ts_utils.py              # Time series utilities
│   └── ...
│
├── *_Python/                     # Individual forecasting templates
│   ├── main.py                  # Template execution script
│   └── config.yaml              # Configuration
│
└── docs/
    ├── sphinx/                      # ReadTheDocs documentation
    └── planning/                    # Reference documents (see docs/planning/README.md)
```

## Data Format

Production data should be CSV files with:

```csv
well_id,date,oil_rate,gas_rate,water_rate,cum_oil,cum_gas
WELL_001,2020-01-01,100.5,50.2,10.1,0,0
WELL_001,2020-02-01,95.3,48.1,9.8,2958,1479
...
```

**Required columns:**
- `well_id`: Unique well identifier
- `date`: Date of measurement (YYYY-MM-DD)
- `oil_rate`: Oil production rate
- `gas_rate`: Gas production rate (optional)
- `water_rate`: Water production rate (optional)

**Optional columns:**
- `cum_oil`, `cum_gas`, `cum_water`: Cumulative production

## Decline Curve Analysis Models

The repository includes three Arps decline curve models:

1. **Exponential Decline** (`b=0`): `q(t) = q_i * exp(-D_i * t)`
2. **Hyperbolic Decline** (`0<b<1`): `q(t) = q_i / (1 + b*D_i*t)^(1/b)`
3. **Harmonic Decline** (`b=1`): `q(t) = q_i / (1 + D_i*t)`

These serve as **baseline comparisons** for time series forecasting methods.

## Evaluation Metrics

All models are evaluated using consistent metrics:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

Models are automatically ranked by performance (RMSE by default).

## Available Time Series Templates

### Classical Methods
- **ARIMA** - Autoregressive Integrated Moving Average
- **ARAR** - Autoregressive Autoregressive
- **BoxJenkins** - Systematic Box-Jenkins methodology
- **MovingAverage** - Simple and exponential moving averages
- **ExponentialSmoothing** - Holt-Winters exponential smoothing
- **VAR** - Vector Autoregression

### Bayesian & Statistical
- **Bayesian** - PyMC Bayesian time series modeling
- **BayesianChangePoint** - Bayesian change point detection
- **Orbit** - Bayesian structural time series
- **PyBSTS** - Bayesian structural time series
- **Nixtla** - StatsForecast fast statistical forecasting

### Modern Forecasting Libraries
- **Prophet** - Facebook's Prophet
- **Greykite** - LinkedIn's forecasting library
- **Darts** - Unified forecasting interface
- **Chronos** - Amazon Chronos transformer
- **TimesFM** - Google TimesFM foundation model
- **LagLlama** - Lag-Llama foundation model
- **StatsForecast** - Nixtla statsforecast AutoARIMA

### Deep Learning
- **LSTM** - Long Short-Term Memory networks
- **NBEATS** - Neural Basis Expansion Analysis
- **TSAI** - Deep learning for time series
- **BERT** - Time series classification

### Feature Engineering & Analysis
- **TSFresh** - Automated feature extraction
- **Aeon** - Time series analysis toolkit
- **Kalman** - State space models
- **Differencing** - Differencing diagnostics
- **IrregularSeries** - Irregular data handling
- **ForecastErrorAnalysis** - Forecast error diagnostics

### Specialized
- **Merlion** - Forecasting & anomaly detection
- **MFLEs** - Multi-frequency learning ensemble
- **Autogluon** - Automated time series forecasting
- **Econometrics** - Causal inference methods
- **CCM** - Convergent Cross Mapping
- **SparseRegression** - LASSO/Ridge/Elastic Net
- **STUMPY_PyOD** - Matrix profile and outlier detection
- **BollingerBands** - Technical indicators
- **SerialCorrelation** - Serial correlation tests
- **ConfidenceIntervals** - Bootstrap intervals
- **RegimeSwitching** - Markov switching models
- **TimeSeriesDecomposition** - Trend/seasonal decomposition
- **tslearn** - Time series machine learning
- **Volatility** - ARCH/GARCH volatility modeling
- **TransferEntropy** - Information-theoretic inference
- **Copula** - Multivariate dependency modeling
- **PyTimeTK** - Feature engineering toolkit
- **OrderedEvaluation** - Ordinal forecast scoring

## ️ Using Individual Templates

Each template can still be used standalone:

1. **Navigate to template directory**:
   ```bash
   cd Prophet_Python
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place data** in the shared `data/` directory

4. **Update `config.yaml`** with your data file name and column names

5. **Run**:
   ```bash
   python main.py
   ```

## Integration with DCA Projects

This repository is designed to connect with decline curve analysis workflows:

1. **Standardized Output**: All forecasts use the same format (pandas Series with datetime index)
2. **Comparable Metrics**: Use the same evaluation metrics as DCA models
3. **Unified API**: Single interface to run both TS and DCA models
4. **Production Data Format**: Standardized CSV format for production rates

## Documentation

- **Integration Plans**: See `docs/planning/` for integration roadmaps and reference materials
- **Data Format**: See `data/production/README.md` for production data specifications
- **Examples**: See `examples/ts_vs_dca_comparison.py` for a complete workflow

## Features

1. **Unified Pipeline**: Run multiple models with one API
2. **DCA Integration**: Compare time series forecasts against traditional DCA models
3. **Standardized Evaluation**: Consistent metrics across all models
4. **Production-Ready**: Designed for real oil & gas production data
5. **Extensible**: Easy to add new models via the registry system
6. **DRY Structure**: Shared utilities eliminate code duplication

## Roadmap

### Phase 1: Core Infrastructure 
- [x] DCA model implementations (Arps exponential, hyperbolic, harmonic)
- [x] Unified forecasting pipeline
- [x] Evaluation framework with standard metrics

### Phase 2: Integration (In Progress)
- [ ] Wrap existing templates for pipeline integration
- [ ] Add more production data examples
- [ ] Enhanced comparison tools (statistical tests, uncertainty quantification)

### Phase 3: Advanced Features (Planned)
- [ ] Automatic model selection
- [ ] Ensemble forecasting
- [ ] Uncertainty quantification (prediction intervals)
- [ ] Multi-well batch processing
- [ ] Web dashboard for model comparison

## Contributing

This is a personal research collection, but feel free to:
- Explore the methodologies
- Adapt code for your projects
- Suggest improvements or report issues

## License

MIT License - See individual files for specific attributions and data source licenses.
