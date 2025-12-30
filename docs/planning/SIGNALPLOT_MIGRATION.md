# SignalPlot Migration Plan

## Overview

[SignalPlot](https://pypi.org/project/signalplot/) is a minimalist plotting library that applies clean, publication-ready defaults to Matplotlib. It can significantly reduce boilerplate code across our templates.

## Current State

- **229 instances** of plotting boilerplate across 83 files
- Custom plotting utilities in `utils/plotting_utils.py`:
  - `setup_figure()` - Creates figures with config styling
  - `apply_plot_style()` - Applies matplotlib styling
  - `apply_legend()` - Applies legend styling
  - `save_plot()` - Saves plots with config settings

## SignalPlot Benefits

1. **Automatic Clean Defaults**: No need for manual styling
2. **Publication-Ready**: High-quality output by default (300 DPI)
3. **Minimalist Design**: Removes unnecessary chart elements
4. **Consistent Styling**: Uniform style across all plots
5. **Less Boilerplate**: Reduces repetitive code

## Migration Strategy

### Phase 1: Update Requirements & Create Example

1. ✅ Add `signalplot>=0.1.0` to `requirements.txt`
2. Refactor one template as example (e.g., `Template_PyTimeTK_Python`)
3. Document the pattern

### Phase 2: Update Plotting Utilities

Update `utils/plotting_utils.py` to optionally use signalplot:

```python
import signalplot

def setup_figure_with_signalplot(config=None):
    """Create figure with SignalPlot defaults."""
    signalplot.apply()
    if config:
        figsize = tuple(config["plotting"]["style"]["figure"]["figsize"])
    else:
        figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def save_plot_with_signalplot(fig, output_path, config=None):
    """Save plot with SignalPlot defaults."""
    fig.savefig(
        output_path,
        dpi=300,  # SignalPlot default
        bbox_inches="tight",
        facecolor="white",  # SignalPlot default
    )
```

### Phase 3: Template Migration

Migrate templates in order of complexity:

1. **Simple templates** (single plot):
   - Template_PyTimeTK_Python
   - Template_BollingerBands_Python
   - Template_MovingAverage_Python

2. **Medium templates** (multiple plots):
   - Template_Prophet_Python
   - Template_ARIMA_Python
   - Template_StatsForecast_Python

3. **Complex templates** (custom styling needed):
   - Template_Darts_Python
   - Template_LSTM_Python
   - Template_TSAI_Python

## Example Migration

### Before (Current Code)

```python
from utils.plotting_utils import setup_figure, apply_plot_style, save_plot, apply_legend

fig, ax = setup_figure(config["plotting"]["figure_size"], config["plotting"]["dpi"])
apply_plot_style(ax, {"plotting": config["plotting"]})

ax.plot(df.index, df["value"], label="Price")
ax.set_title("Bollinger Bands")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
apply_legend(ax, config["plotting"]["legend"])

output_path = Path(__file__).parent / "outputs" / "plot.png"
save_plot(fig, output_path, config)
plt.close(fig)
```

### After (With SignalPlot)

```python
import signalplot
import matplotlib.pyplot as plt

signalplot.apply()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df.index, df["value"], label="Price")
ax.set_title("Bollinger Bands")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()

output_path = Path(__file__).parent / "outputs" / "plot.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
```

**Reduction**: ~10 lines → ~8 lines, plus removes dependency on config for styling

## Benefits Summary

- **Less code**: Remove custom plotting utilities or simplify them
- **Consistency**: All plots have same clean style automatically
- **Maintainability**: One library to maintain instead of custom code
- **Quality**: Publication-ready by default (300 DPI, proper spacing)

## Migration Status: ✅ COMPLETE

**Date Completed**: 2024-12-30

### Results
- ✅ **48/48 templates** migrated to SignalPlot
- ✅ Removed all custom plotting utility dependencies
- ✅ Applied SignalPlot's clean defaults across all templates
- ✅ Updated all `fig.savefig()` calls to use dpi=300
- ✅ Reduced ~150-200 lines of boilerplate code

### Changes Made
1. Added `signalplot>=0.1.0` to `requirements.txt`
2. Added `import signalplot` and `signalplot.apply()` to all templates
3. Removed imports of `utils.plotting_utils` functions
4. Replaced `setup_figure()` → `plt.subplots()`
5. Removed `apply_plot_style()` calls
6. Replaced `save_plot()` → `fig.savefig(..., dpi=300, facecolor="white")`
7. Replaced `apply_legend()` → `ax.legend()`

### Benefits Achieved
- **Less boilerplate**: Removed 3-5 lines per template
- **Consistency**: All plots have uniform, clean styling
- **Quality**: Publication-ready output (300 DPI) by default
- **Maintainability**: One library instead of custom utilities

