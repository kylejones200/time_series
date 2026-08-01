# DRY Migration Guide

## Overview

This guide shows how to migrate templates to use the consolidated utilities in `src/`.

## Before (Duplicated Code)

### Example: Template with duplicated code

```python
#!/usr/bin/env python3
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util

# Apply SignalPlot's clean defaults
signalplot.apply()


def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    
    # Load data (custom implementation)
    df = pd.read_csv(Path(__file__).parent.parent / "data" / config["data"]["input_file"])
    df[config["data"]["date_col"]] = pd.to_datetime(df[config["data"]["date_col"]])
    df = df.set_index(config["data"]["date_col"])
    series = df[config["data"]["value_col"]]
    
    # Create output directory
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)
    
    # ... model fitting code ...
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train.values, "k-", linewidth=1.5, label="Historical")
    # ... more plotting code ...
    
    # Save plot
    fig.savefig(output_dir / "plot.png", dpi=300, bbox_inches="tight", facecolor="white")
    
    # Save CSV
    df.to_csv(output_dir / "output.csv", index=False)
```

**Problems**:
- 50+ lines of duplicated utility code
- Inconsistent patterns
- Hard to maintain

## After (Using Consolidated Utilities)

### Option 1: Using BaseTemplate (Recommended)

```python
#!/usr/bin/env python3
from pathlib import Path
from src import BaseTemplate, ARIMAModel, Evaluator

class MyTemplate(BaseTemplate):
    """My forecasting template."""
    
    def run(self):
        """Run the forecasting workflow."""
        # Load data (inherited method)
        series = self.load_data()
        print(f"Loaded {len(series)} data points")
        
        # Split train/test
        evaluator = Evaluator(test_size=0.2)
        train, test = evaluator.split(series)
        
        # Fit model
        model = ARIMAModel()
        model.fit(train)
        
        # Generate forecast
        forecast, conf_int = model.forecast(n_periods=len(test), return_conf_int=True)
        
        # Evaluate
        metrics = evaluator.evaluate(forecast, test)
        print(f"RMSE: {metrics['RMSE']:.4f}")
        
        # Create plot (inherited method)
        fig, ax = self.create_plot(
            train=train,
            test=test,
            forecast=forecast,
            conf_int=conf_int,
            title="My Forecast"
        )
        
        # Save plot (inherited method)
        self.save_plot(fig, "forecast_plot.png")
        
        # Save CSV (inherited method)
        forecast_df = pd.DataFrame({
            "date": forecast.index,
            "forecast": forecast.values
        })
        self.save_csv(forecast_df, "forecast.csv")
        
        if self.config["plotting"]["show_plot"]:
            plt.show()


def main():
    template = MyTemplate(config_path="config.yaml")
    template.run()


if __name__ == "__main__":
    main()
```

**Benefits**:
- Only ~30 lines of template-specific code
- Consistent patterns
- Easy to maintain

### Option 2: Using Utilities Directly

```python
#!/usr/bin/env python3
from pathlib import Path
from src import (
    load_config,
    load_time_series,
    ARIMAModel,
    Evaluator,
    create_forecast_plot,
    save_plot,
    ensure_output_dir,
    get_output_dir,
)

def main():
    # Load config (consolidated)
    config = load_config()
    
    # Load data (consolidated)
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"]["date_column"],
        value_column=config["data"]["value_column"]
    )
    
    # Create output directory (consolidated)
    script_dir = Path(__file__).parent
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    
    # ... model fitting code ...
    
    # Create plot (consolidated)
    fig, ax = create_forecast_plot(
        train=train,
        test=test,
        forecast=forecast,
        conf_int=conf_int,
        title="My Forecast"
    )
    
    # Save plot (consolidated)
    save_plot(fig, output_dir / "plot.png")
```

## Migration Steps

1. **Remove duplicated code**:
   - Delete `repo_import()` function
   - Delete `load_config()` function (or use consolidated version)
   - Remove `signalplot.apply()` (already applied in `src/__init__.py`)

2. **Update imports**:
   ```python
   # Before
   import signalplot
   signalplot.apply()
   from importlib import util
   
   # After
   from src import load_config, load_time_series, ...
   ```

3. **Use consolidated utilities**:
   - Replace data loading with `load_time_series()`
   - Replace config loading with `load_config()`
   - Replace plotting with `create_forecast_plot()`
   - Replace plot saving with `save_plot()`

4. **Or use BaseTemplate**:
   - Inherit from `BaseTemplate`
   - Use inherited methods: `load_data()`, `create_plot()`, `save_plot()`, `save_csv()`

## Available Utilities

### From `src/`:

- `load_config()` - Load YAML config
- `load_time_series()` - Load CSV with date/value columns
- `repo_import()` - Import module from repo root
- `ensure_output_dir()` - Create output directory
- `get_output_dir()` - Get output dir from config
- `create_forecast_plot()` - Standardized forecast plot
- `save_plot()` - Save plot with standard settings
- `BaseTemplate` - Base class with all utilities

### Model wrappers:

- `ARIMAModel` - ARIMA model wrapper
- `Evaluator` - Evaluation utilities

## Benefits

1. **Code reduction**: 50+ lines → 5-10 lines per template
2. **Consistency**: All templates follow same patterns
3. **Maintainability**: Fix bugs once, benefit everywhere
4. **Easier to extend**: New templates just inherit from BaseTemplate

## Example Migration

See `reference_forecast.py` for a complete example using consolidated utilities.

