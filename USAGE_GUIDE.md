# Usage Guide

This guide shows you the easiest ways to use the time series forecasting library.

## Three Ways to Get Started

### 1. Interactive Wizard (Easiest)

Perfect for first-time users:

```bash
python scripts/quick_start.py
```

The wizard will:
-  Help you find and validate your data
-  Map your columns if needed
-  Recommend the best templates
-  Run your first forecast
-  Show you next steps

### 2. Unified CLI (Most Flexible)

Run any template from the root directory:

```bash
# See all available templates
python forecast.py list

# Validate your data format
python forecast.py validate data/my_series.csv

# Get intelligent recommendations
python forecast.py recommend data/my_series.csv

# Run a specific template
python forecast.py run ARIMA_Python --data data/my_series.csv

# Compare multiple templates
python forecast.py benchmark --data data/my_series.csv
```

### 3. Direct Template Execution (Traditional)

Navigate to a template and run it:

```bash
cd ARIMA_Python
python main.py
```

## Common Workflows

### Workflow 1: Quick Forecast

You have data and want a quick forecast:

```bash
# 1. Validate your data
python forecast.py validate data/my_series.csv

# 2. Get recommendations
python forecast.py recommend data/my_series.csv

# 3. Run the top recommendation
python forecast.py run ARIMA_Python --data data/my_series.csv
```

### Workflow 2: Compare Multiple Methods

You want to find the best method for your data:

```bash
# Benchmark fast methods
python forecast.py benchmark --data data/my_series.csv --categories fast

# Or specify templates
python forecast.py benchmark --data data/my_series.csv --templates ARIMA_Python Prophet_Python StatsForecast_Python

# Compare results
python scripts/compare_results.py --output-dir outputs/benchmark
```

### Workflow 3: Production Data Analysis

You have well production data:

```bash
# 1. Generate example data (if needed)
python data/production/generate_example_data.py

# 2. Run comparison with DCA models
python examples/ts_vs_dca_comparison.py

# 3. Or use the pipeline programmatically
python -c "
from pipelines import ForecastingPipeline
from models.dca import ArpsExponential

pipeline = ForecastingPipeline(
    data_path='data/production/well_production.csv',
    target_column='oil_rate',
    forecast_horizon=12
)
pipeline.add_model('Arps Exponential', ArpsExponential())
results = pipeline.run_all()
print(pipeline.compare_models(results))
"
```

## ️ Available Tools

### Data Validation

```bash
python scripts/data_validator.py data/my_series.csv
```

Checks:
- File format and readability
- Required columns (date, value)
- Data quality (missing values, outliers)
- Data length and characteristics
- Provides fix suggestions

### Model Selection

```bash
python scripts/model_selector.py data/my_series.csv
```

Analyzes:
- Data length
- Trend presence
- Seasonality
- Volatility
- Stationarity

Recommends:
- Top 10 templates with confidence scores
- Reasons for each recommendation
- Next steps

### Automated Benchmarking

```bash
python scripts/auto_benchmark.py --data data/my_series.csv --categories fast
```

Runs multiple templates and generates:
- Comparison CSV
- Summary JSON
- Individual outputs

### Result Comparison

```bash
python scripts/compare_results.py --output-dir outputs/benchmark --actual data/test_values.csv
```

Creates:
- Visual comparison plots
- Summary tables with metrics
- Error analysis

## Tips

1. **Start with validation** - Always validate your data first
2. **Use recommendations** - Let the tool suggest templates based on your data
3. **Benchmark first** - Compare a few fast methods before running expensive ones
4. **Check outputs** - Review plots and metrics in `outputs/` directories
5. **Read template READMEs** - Each template has specific usage notes

## 🆘 Getting Help

- **Data issues?** Run `python forecast.py validate <your_data.csv>`
- **Don't know which template?** Run `python forecast.py recommend <your_data.csv>`
- **First time?** Run `python scripts/quick_start.py`
- **Need examples?** Check `examples/` directory
- **Full documentation?** See `docs/sphinx/` or ReadTheDocs

## Next Steps

- Explore templates: `python forecast.py list`
- Read template READMEs in each `*_Python/` directory
- Check out examples in `examples/`
- Review full documentation in `docs/sphinx/`

