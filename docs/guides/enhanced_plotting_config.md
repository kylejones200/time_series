# Enhanced Plotting Configuration Guide

## Overview

This guide shows how to use the enhanced plotting configuration structure for more granular control over plot styling. This is an **optional enhancement** - templates can use either the simple config structure or the enhanced one.

## Quick Comparison

### Simple Config (Current Default)

```yaml
plotting:
  figure_size: [12, 6]
  dpi: 150
  show_plot: true
```

### Enhanced Config (Optional)

```yaml
plotting:
  style:
    spines:
      top: false
      right: false
      bottom: true
      left: true
    grid: false
    legend:
      frameon: false
      loc: "best"
    figure:
      figsize: [12, 6]
      dpi: 150
    colors:
      primary: "k"
      secondary: "r"
      accent: "b"
      background: "w"
    linewidth: 1.5
    markersize: 4
    alpha: 0.7
  show_plot: true
```

## Using Enhanced Config

### Step 1: Update Your Config File

Add the enhanced `plotting.style` section to your `config.yaml`:

```yaml
plotting:
  style:
    spines:
      top: false      # Hide top spine
      right: false    # Hide right spine
      bottom: true    # Show bottom spine
      left: true      # Show left spine
    grid: false       # Show/hide grid
    legend:
      frameon: false # Legend without frame
      loc: "best"     # Legend location
    figure:
      figsize: [12, 6]
      dpi: 150
    colors:
      primary: "k"    # Primary color (black)
      secondary: "r"  # Secondary color (red)
      accent: "b"     # Accent color (blue)
      background: "w" # Background color (white)
    linewidth: 1.5
    markersize: 4
    alpha: 0.7
  show_plot: true
```

### Step 2: Use `utils.plotting_utils` Instead of `src.plotting`

In your template's `main.py`, import from `utils.plotting_utils`:

```python
from utils.plotting_utils import (
    apply_plot_style,
    setup_figure,
    apply_legend,
    save_plot as save_plot_enhanced,
)
```

### Step 3: Update Your Plotting Code

Replace your plotting code to use the enhanced utilities:

```python
def create_visualizations(results, config):
    """Generate clean, minimalist visualizations."""
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(exist_ok=True)

    # Use setup_figure instead of plt.subplots
    fig, ax = setup_figure(config, nrows=1, ncols=1)
    
    # Your plotting code here
    ax.plot(
        results.index,
        results.values,
        color=config["plotting"]["style"]["colors"]["primary"],
        linewidth=config["plotting"]["style"]["linewidth"],
        alpha=config["plotting"]["style"]["alpha"],
        label="Data"
    )
    
    # Apply legend styling
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    # Save with enhanced save_plot
    if config["output"]["save_plots"]:
        save_plot_enhanced(
            fig,
            output_dir / "results.png",
            config
        )
    
    if config["plotting"]["show_plot"]:
        plt.show()
    else:
        plt.close(fig)
```

## Configuration Options

### Spines

Control which plot borders (spines) are visible:

```yaml
spines:
  top: false    # Hide top border
  right: false  # Hide right border
  bottom: true  # Show bottom border
  left: true    # Show left border
```

**Common patterns:**
- **Minimalist**: `top: false, right: false` (removes top and right)
- **Full border**: All `true`
- **No border**: All `false`

### Grid

```yaml
grid: false  # true or false
```

When `true`, shows a grid with 30% opacity.

### Legend

```yaml
legend:
  frameon: false  # Legend without frame (cleaner look)
  loc: "best"     # Location: "best", "upper right", "lower left", etc.
```

**Common locations:**
- `"best"` - Automatic best location
- `"upper right"`, `"upper left"`, `"lower right"`, `"lower left"`
- `"center"`, `"center left"`, `"center right"`

### Figure

```yaml
figure:
  figsize: [12, 6]  # [width, height] in inches
  dpi: 150          # Resolution for saved figures
```

### Colors

Define a color palette for your plots:

```yaml
colors:
  primary: "k"      # Black
  secondary: "r"    # Red
  accent: "b"       # Blue
  background: "w"   # White
```

**Color options:**
- Single letters: `"k"` (black), `"r"` (red), `"b"` (blue), `"g"` (green), `"c"` (cyan), `"m"` (magenta), `"y"` (yellow)
- Hex codes: `"#1f77b4"`, `"#ff7f0e"`, etc.
- Named colors: `"navy"`, `"crimson"`, `"forestgreen"`, etc.

### Line and Marker Styling

```yaml
linewidth: 1.5    # Line width for plots
markersize: 4     # Marker size for scatter plots
alpha: 0.7        # Transparency (0.0 to 1.0)
```

## Example: Complete Template

Here's a complete example of a template using enhanced config:

```python
#!/usr/bin/env python3
"""
Example template using enhanced plotting configuration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
from utils.plotting_utils import (
    setup_figure,
    apply_plot_style,
    apply_legend,
    save_plot as save_plot_enhanced,
)
from src.config import load_config
from src.loader import load_time_series


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    config = load_config()
    
    # Load data
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"]["date_col"],
        value_column=config["data"]["value_col"]
    )
    
    # Create visualization with enhanced config
    fig, ax = setup_figure(config, nrows=1, ncols=1)
    
    # Plot data using config colors
    style = config["plotting"]["style"]
    ax.plot(
        series.index,
        series.values,
        color=style["colors"]["primary"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        label="Time Series"
    )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("Enhanced Config Example")
    
    # Apply legend with config settings
    apply_legend(ax, config)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    if config["output"].get("save_plots", True):
        save_plot_enhanced(
            fig,
            output_dir / "enhanced_plot.png",
            config
        )
    
    if config["plotting"]["show_plot"]:
        plt.show()
    else:
        plt.close(fig)
    
    print(" Analysis complete")


if __name__ == "__main__":
    main()
```

## Migration Guide

### From Simple Config to Enhanced Config

1. **Update config.yaml**: Add the `plotting.style` section
2. **Update imports**: Change from `src.plotting` to `utils.plotting_utils`
3. **Update plotting code**: Use `setup_figure()`, `apply_plot_style()`, `apply_legend()`
4. **Update save_plot**: Use `save_plot()` from `utils.plotting_utils` (takes `config` parameter)

### Backward Compatibility

Templates using the simple config structure will continue to work. The enhanced config is **optional** and templates can choose which approach to use.

## Benefits

 **Consistent styling** across all plots  
 **Centralized configuration** - change styles in one place  
 **Professional appearance** - clean, minimalist plots  
 **Easy customization** - adjust colors, linewidths, etc. via config  
 **Reusable** - same config structure across multiple templates  

## See Also

- `utils/plotting_utils.py` - Implementation of enhanced plotting utilities
- `docs/examples/enhanced_config_example.yaml` - Example enhanced config file
- `docs/examples/enhanced_config_template_example.py` - Working example template
- `src/plotting.py` - Simple plotting utilities (alternative approach)

