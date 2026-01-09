# Decline Curve Analysis Integration Plan

## Vision

Transform the time_series repository from a collection of templates into a **production forecasting pipeline** that compares time series forecasting methods against traditional decline curve analysis (DCA) models.

## Goals

1. **Forecasting Pipeline**: Each template produces concrete forecasts for oil/gas/water production rates
2. **DCA Integration**: Decline curve models (Arps, hyperbolic, exponential) as baseline comparisons
3. **Unified API**: Single interface to run multiple models and compare results
4. **Standardized Evaluation**: Consistent metrics (MSE, MAE, MAPE, R²) across all models
5. **Production-Ready Output**: CSV results, comparison plots, and model selection recommendations

## Architecture

### Proposed Structure

```
time_series/
├── pipelines/                    # NEW: Unified forecasting pipelines
│   ├── __init__.py
│   ├── forecasting_pipeline.py  # Main pipeline orchestrator
│   ├── model_registry.py        # Registry of all available models
│   └── evaluator.py             # Model evaluation and comparison
│
├── models/                      # NEW: Model implementations
│   ├── __init__.py
│   ├── dca/                     # Decline Curve Analysis models
│   │   ├── __init__.py
│   │   ├── arps.py              # Arps models (exponential, hyperbolic)
│   │   ├── exponential.py       # Exponential decline
│   │   └── hyperbolic.py        # Hyperbolic decline
│   └── forecasting/             # Time series forecasting models
│       ├── __init__.py
│       ├── arima_wrapper.py
│       ├── prophet_wrapper.py
│       └── ...                  # Wrappers for existing templates
│
├── data/                        # Existing
│   ├── production/              # NEW: Production data examples
│   │   ├── well_production.csv  # Example well data
│   │   ├── multi_well.csv       # Multiple wells
│   │   └── README.md
│   └── ...
│
├── evaluation/                  # NEW: Evaluation and comparison tools
│   ├── __init__.py
│   ├── metrics.py               # Standard metrics (MSE, MAE, MAPE, etc.)
│   ├── comparison.py            # Model comparison utilities
│   └── reporting.py             # Generate comparison reports
│
├── utils/                       # Existing (enhance)
│   ├── ts_utils.py
│   ├── plotting_utils.py
│   └── dca_utils.py             # NEW: DCA-specific utilities
│
├── *_Python/                    # Existing templates (wrap in pipeline)
│   └── ...
│
└── examples/                    # NEW: Complete examples
    ├── single_well_forecast.py
    ├── multi_well_comparison.py
    └── ts_vs_dca_comparison.py
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

1. **Create DCA Model Implementations**
   - Arps exponential decline: `q(t) = q_i * exp(-D_i * t)`
   - Arps hyperbolic decline: `q(t) = q_i / (1 + b * D_i * t)^(1/b)`
   - Arps harmonic decline: `q(t) = q_i / (1 + D_i * t)`
   - Fit models using least squares or MLE

2. **Create Unified Pipeline API**
   - `ForecastingPipeline` class
   - Model registry system
   - Standardized input/output format

3. **Create Evaluation Framework**
   - Standard metrics (MSE, MAE, MAPE, R²)
   - Rolling origin evaluation
   - Model comparison utilities

### Phase 2: Integration (Week 2-3)

1. **Wrap Existing Templates**
   - Create adapter functions for each template
   - Standardize inputs/outputs
   - Integrate with pipeline

2. **Create Production Data Examples**
   - Standardized CSV format
   - Multiple well examples
   - Synthetic decline curve data

3. **Comparison Tools**
   - Side-by-side forecast plots
   - Model performance tables
   - Statistical comparison tests

### Phase 3: Examples and Documentation (Week 3-4)

1. **Complete Examples**
   - Single well forecasting
   - Multi-well comparison
   - TS vs DCA comparison

2. **Documentation**
   - Updated README with new workflow
   - API documentation
   - Example notebooks

## Data Format

### Production Data CSV

```csv
well_id,date,oil_rate,gas_rate,water_rate,cum_oil,cum_gas
WELL_001,2020-01-01,100.5,50.2,10.1,0,0
WELL_001,2020-02-01,95.3,48.1,9.8,2958,1479
...
```

Required columns:
- `well_id`: Unique well identifier
- `date`: Date of measurement
- `*_rate`: Production rates (oil, gas, water)
- `cum_*`: Cumulative production (optional)

## API Design

### Example Usage

```python
from pipelines import ForecastingPipeline
from models.dca import ArpsExponential, ArpsHyperbolic
from models.forecasting import ARIMAWrapper, ProphetWrapper

# Initialize pipeline
pipeline = ForecastingPipeline(
    data_path="data/production/well_production.csv",
    well_id="WELL_001",
    target_column="oil_rate",
    forecast_horizon=12  # months
)

# Add models
pipeline.add_model("Arps Exponential", ArpsExponential())
pipeline.add_model("Arps Hyperbolic", ArpsHyperbolic())
pipeline.add_model("ARIMA", ARIMAWrapper())
pipeline.add_model("Prophet", ProphetWrapper())

# Run all models
results = pipeline.run_all()

# Evaluate and compare
comparison = pipeline.compare_models(results)

# Save results
pipeline.save_results(results, output_dir="outputs/well_001/")
pipeline.generate_report(comparison, output_path="outputs/well_001/comparison_report.html")
```

## Model Output Format

All models should return:
- `forecast`: Forecasted values (pandas Series with datetime index)
- `confidence_intervals`: Upper and lower bounds (optional)
- `model_params`: Fitted parameters
- `metrics`: Evaluation metrics (if test data provided)

## Evaluation Metrics

Standard metrics for comparison:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination
- **AIC/BIC**: Information criteria (for model selection)

## Next Steps

1. Implement Phase 1 (DCA models + pipeline infrastructure)
2. Create example production dataset
3. Build wrapper for one template as proof of concept
4. Create comparison example

