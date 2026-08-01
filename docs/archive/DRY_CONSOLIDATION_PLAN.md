# DRY Consolidation Plan

## Current Duplication Issues

### 1. **`repo_import()` Function** (48+ instances)
Every template has an identical `repo_import()` function:
```python
def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj
```

**Solution**: Move to `src/utils.py` and import directly.

### 2. **Config Loading** (48+ variations)
- Some use `yaml.safe_load()` directly
- Some use dataclasses with `@dataclass`
- Some parse config into custom Config objects
- Inconsistent path handling

**Solution**: Create `src/config.py` with unified config loader.

### 3. **Data Loading** (Inconsistent)
- Some use `utils.ts_utils.load_ts_data()`
- Some do direct pandas loading
- Inconsistent column name handling

**Solution**: All should use `src/loader.py` (already created).

### 4. **Output Directory Creation** (Repeated pattern)
```python
output_dir = Path(__file__).parent / config["output"]["output_dir"]
output_dir.mkdir(exist_ok=True)
```

**Solution**: Add to `src/utils.py`.

### 5. **Plot Saving** (Repeated pattern)
```python
fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
```

**Solution**: Create `src/plotting.py` helper.

### 6. **Signalplot Application** (48+ instances)
At least this is consistent, but still duplicated.

**Solution**: Move to shared initialization in `src/__init__.py`.

## Consolidation Strategy

### Phase 1: Create Shared Utilities (`src/utils.py`)

```python
# src/utils.py
from pathlib import Path
from importlib import util

def repo_import(module: str):
    """Import module from repository root."""
    # ... (consolidate all instances)

def ensure_output_dir(output_dir: Path) -> Path:
    """Ensure output directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
```

### Phase 2: Create Config Loader (`src/config.py`)

```python
# src/config.py
import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Normalize paths relative to config file location
    config_dir = Path(config_path).parent
    if "data" in config and "input_file" in config["data"]:
        config["data"]["input_file"] = str(config_dir.parent / "data" / config["data"]["input_file"])
    
    return config

def get_output_dir(config: Dict[str, Any], script_dir: Path) -> Path:
    """Get output directory from config."""
    output_dir_name = config.get("output", {}).get("output_dir", "outputs")
    return script_dir / output_dir_name
```

### Phase 3: Create Base Template Class (`src/base_template.py`)

```python
# src/base_template.py
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import matplotlib.pyplot as plt

class BaseTemplate:
    """Base class for all forecasting templates."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.script_dir = self.config_path.parent
        self.config = load_config(config_path)
        self.output_dir = ensure_output_dir(
            get_output_dir(self.config, self.script_dir)
        )
        
    def load_data(self) -> pd.Series:
        """Load time series data using standard loader."""
        from .loader import load_time_series
        return load_time_series(
            self.config["data"]["input_file"],
            date_column=self.config["data"].get("date_column", "date"),
            value_column=self.config["data"].get("value_column", "value")
        )
    
    def save_plot(self, fig, filename: str):
        """Save plot with standard settings."""
        path = self.output_dir / filename
        fig.savefig(
            path,
            dpi=self.config.get("output", {}).get("dpi", 300),
            bbox_inches="tight",
            facecolor="white"
        )
        return path
    
    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to CSV."""
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return path
```

### Phase 4: Create Plotting Utilities (`src/plotting.py`)

```python
# src/plotting.py
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple

def create_forecast_plot(
    train: pd.Series,
    test: Optional[pd.Series],
    forecast: pd.Series,
    conf_int: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Forecast"
) -> Tuple[plt.Figure, plt.Axes]:
    """Create standardized forecast plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot historical
    ax.plot(train.index, train.values, "k-", linewidth=1.5, label="Historical", alpha=0.8)
    
    # Plot test if provided
    if test is not None:
        ax.plot(test.index, test.values, "g-", linewidth=1.5, label="Actual", alpha=0.8)
    
    # Plot forecast
    ax.plot(forecast.index, forecast.values, "r--", linewidth=1.5, label="Forecast", alpha=0.8)
    
    # Plot confidence intervals if provided
    if conf_int is not None:
        ax.fill_between(
            conf_int.index,
            conf_int["lower"],
            conf_int["upper"],
            color="r",
            alpha=0.2,
            label="95% CI"
        )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    return fig, ax
```

### Phase 5: Update `src/__init__.py` to Initialize Signalplot

```python
# src/__init__.py
import signalplot

# Apply SignalPlot's clean defaults once at module level
signalplot.apply()

from .loader import load_time_series
from .model import ARIMAModel
from .evaluator import Evaluator
from .config import load_config, get_output_dir
from .utils import repo_import, ensure_output_dir
from .plotting import create_forecast_plot
from .base_template import BaseTemplate

__all__ = [
    "load_time_series",
    "ARIMAModel",
    "Evaluator",
    "load_config",
    "get_output_dir",
    "repo_import",
    "ensure_output_dir",
    "create_forecast_plot",
    "BaseTemplate",
]
```

## Migration Strategy

### Step 1: Create consolidated utilities
-  `src/utils.py` - Common utilities
-  `src/config.py` - Unified config loading
-  `src/plotting.py` - Plotting helpers
-  `src/base_template.py` - Base template class

### Step 2: Update reference script to use new utilities
- Refactor `reference_forecast.py` to use consolidated utilities

### Step 3: Migrate templates incrementally
- Start with 2-3 templates as examples
- Update others to follow the same pattern
- All templates should inherit from `BaseTemplate` or use the utilities

### Step 4: Remove duplicated code
- Delete `repo_import()` from all templates
- Remove redundant config loading code
- Consolidate data loading to use `src/loader.py`

## Benefits

1. **Single source of truth** for common functionality
2. **Easier maintenance** - fix bugs once, benefit everywhere
3. **Consistency** - all templates follow same patterns
4. **Easier to extend** - new templates just inherit from BaseTemplate
5. **Smaller codebase** - eliminate thousands of lines of duplicated code

## Example: Before vs After

### Before (Each template):
```python
# 48 templates × 50 lines = 2400 lines of duplicated code

def repo_import(module: str):
    # ... 10 lines

def load_config(config_path="config.yaml"):
    # ... 5 lines

output_dir = Path(__file__).parent / config["output"]["output_dir"]
output_dir.mkdir(exist_ok=True)
# ... more duplication
```

### After (All templates):
```python
# Single source: src/base_template.py (100 lines)
from src import BaseTemplate

class MyTemplate(BaseTemplate):
    def run(self):
        data = self.load_data()  # Uses shared loader
        # ... template-specific code
        self.save_plot(fig, "output.png")  # Uses shared save
```

**Result**: 2400 lines → 100 lines (96% reduction)

