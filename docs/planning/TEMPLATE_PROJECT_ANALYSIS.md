# Template Project Analysis

## Overview

Analysis of `WIP/template_project/` to determine what can be extracted to the main repository.

## Files in template_project

1. `plotting_utils.py` - Config-driven plotting utilities
2. `main.py` - Template structure for new projects
3. `config.yaml` - Enhanced configuration structure
4. `README.md` - Generic template README
5. `requirements.txt` - Basic dependencies

## Analysis Results

### Already in Main Repo

**`plotting_utils.py`** → **`utils/plotting_utils.py`**
- **Status**: IDENTICAL
- The file already exists in `utils/plotting_utils.py` with the exact same content
- Functions included:
  - `apply_plot_style(ax, config)` - Apply config-driven styling
  - `setup_figure(config, nrows, ncols)` - Create styled figure
  - `apply_legend(ax, config, **kwargs)` - Apply legend styling
  - `_normalize_axes(axes, nrows, ncols)` - Helper for axes normalization
  - `save_plot(fig, output_path, config)` - Save with config settings

**Note**: However, these utilities are **not currently used** in the main repo. Templates use `src/plotting.py` instead, which has simpler functions.

### Potentially Useful

**1. Enhanced Config Structure (`config.yaml`)**

The template_project config has a more detailed plotting structure:

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
```

**Current templates** use simpler structure:
```yaml
plotting:
  figure_size: [12, 6]
  dpi: 150
  show_plot: true
```

**Recommendation**: 
- The enhanced config structure could be useful for templates that want more control
- Could be added as an optional enhancement
- Would require templates to use `utils/plotting_utils.py` instead of `src/plotting.py`

**2. Template Structure (`main.py`)**

The `main.py` provides a clean skeleton:
- `load_config()` - Config loading
- `load_data()` - Data loading with preprocessing
- `run_analysis()` - Placeholder for analysis
- `create_visualizations()` - Visualization creation
- `main()` - Main execution

**Current repo** has:
- `src/base_template.py` - BaseTemplate class (more OOP approach)
- `reference_forecast.py` - Reference implementation

**Recommendation**:
- The template_project structure is simpler (functional vs OOP)
- Could be useful as a reference for creating new templates
- Not necessary to extract since we have `BaseTemplate` and `reference_forecast.py`

### Not Needed

**1. `requirements.txt`**
- Generic dependencies
- Main repo already has comprehensive `requirements.txt`

**2. `README.md`**
- Generic template README
- Not specific to this repository

## Recommendations

### Option 1: Extract Enhanced Config Structure (Recommended)

**Action**: Create a documentation file showing how to use the enhanced config structure with `utils/plotting_utils.py`

**Benefits**:
- Provides templates with more styling control
- Uses existing `utils/plotting_utils.py` code
- Optional enhancement (templates can choose simple or advanced)

**Files to create**:
- `docs/guides/enhanced_plotting_config.md` - Guide on using enhanced config
- Update `utils/plotting_utils.py` to ensure it's properly documented

### Option 2: Delete template_project (Not Recommended)

**Why not**: The enhanced config structure is valuable and could be useful for future templates.

### Option 3: Keep as Reference (Current State)

**Action**: Keep `WIP/template_project/` as a reference for:
- Enhanced config structure
- Alternative template structure (functional vs OOP)

## Conclusion

**What to extract**:
1.  **Enhanced config structure** - Document as optional enhancement
2.  **Usage guide** - Show how to use `utils/plotting_utils.py` with enhanced config

**What NOT to extract**:
1.  `plotting_utils.py` - Already exists identically in `utils/`
2.  `main.py` - We have better alternatives (`BaseTemplate`, `reference_forecast.py`)
3.  `README.md` - Generic, not useful
4.  `requirements.txt` - Already have comprehensive version

## Implementation Status

 **Completed:**

1.  Created `docs/guides/enhanced_plotting_config.md` - Comprehensive guide
2.  Created `docs/examples/enhanced_config_example.yaml` - Example config file
3.  Created `docs/examples/enhanced_config_template_example.py` - Working example template
4.  Documented all configuration options and usage patterns

**Files Created:**
- `docs/guides/enhanced_plotting_config.md` - Full guide with examples
- `docs/examples/enhanced_config_example.yaml` - Example config
- `docs/examples/enhanced_config_template_example.py` - Example template

**Next Steps (Optional):**
- Update a few existing templates to use enhanced config as examples
- Keep `WIP/template_project/` as reference or delete after documenting

