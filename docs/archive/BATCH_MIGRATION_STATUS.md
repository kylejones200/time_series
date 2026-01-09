# Batch Migration Status

## Progress Overview

**Total Templates:** 48
**Already Migrated:** 8 (Prophet, MovingAverage, ExponentialSmoothing, ARAR, BoxJenkins, Differencing, Kalman, Nixtla)
**Remaining:** 40

## Migration Pattern

All templates follow this standard pattern after migration:

```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
    create_forecast_plot,  # when applicable
)
from src.evaluator import Evaluator  # when applicable
```

## Completed Batches

### Batch 1: Simple Templates (COMPLETE)
1.  ARAR_Python
2.  BoxJenkins_Python
3.  Differencing_Python
4.  Kalman_Python
5.  Nixtla_Python

### Previously Migrated
1.  Prophet_Python
2.  MovingAverage_Python
3.  ExponentialSmoothing_Python

## Remaining Batches

### Batch 2: Statistical Templates (TODO)
- Bayesian_Python
- ConfidenceIntervals_Python
- SerialCorrelation_Python
- VAR_Python

### Batch 3: Feature Engineering (TODO)
- TSFresh_Python
- PyTimeTK_Python
- TimeSeriesDecomposition_Python
- IrregularSeries_Python

### Batch 4: Deep Learning (TODO)
- LSTM_Python
- NBEATS_Python
- TSAI_Python
- BERT_Python

### Batch 5: Modern Forecasting (TODO)
- Chronos_Python
- TimesFM_Python
- LagLlama_Python
- Sundial_Python
- Orbit_Python

### Batch 6: Specialized (TODO)
- CCM_Python
- Copula_Python
- RegimeSwitching_Python
- TransferEntropy_Python
- Volatility_Python

### Batch 7: Remaining (TODO)
- Autogluon_Python
- Darts_Python
- Econometrics_Python
- Greykite_Python
- Merlion_Python
- MFLEs_Python
- PyBSTS_Python
- PyCaret_Python
- SparseRegression_Python
- StatsForecast_Python
- STUMPY_PyOD_Python
- BollingerBands_Python
- BayesianChangePoint_Python
- ForecastErrorAnalysis_Python
- OrderedEvaluation_Python
- tslearn_Python
- Aeon_Python

## Migration Checklist (per template)

- [ ] Remove `repo_import()` function
- [ ] Remove `signalplot.apply()` call
- [ ] Replace custom `load_config()` with `src.load_config()`
- [ ] Replace custom data loading with `src.load_time_series()` where applicable
- [ ] Replace custom output dir creation with `src.ensure_output_dir()` and `src.get_output_dir()`
- [ ] Replace custom plotting with `src.create_forecast_plot()` where applicable
- [ ] Replace custom plot saving with `src.save_plot()`
- [ ] Add `sys.path.insert()` at top for src imports
- [ ] Test template still works correctly
- [ ] Update README if needed

## Standard Replacements

### Before:
```python
import signalplot
signalplot.apply()

def repo_import(module: str): ...

def load_config(config_path="config.yaml"): ...

output_dir = Path(__file__).parent / config["output"]["output_dir"]
output_dir.mkdir(exist_ok=True)

fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
```

### After:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_config, load_time_series, ensure_output_dir, get_output_dir, save_plot

script_dir = Path(__file__).parent
config = load_config()
output_dir = ensure_output_dir(get_output_dir(config, script_dir))

save_plot(fig, output_dir / "plot.png", dpi=300)
```

## Benefits So Far

- **Code Reduction**: ~150+ lines eliminated per batch
- **Consistency**: All migrated templates follow same patterns
- **Maintainability**: Fix bugs once, benefit everywhere
- **Clarity**: Less boilerplate, more focus on template-specific logic

## Next Steps

Continue with Batch 2 (Statistical Templates) using the established pattern.

