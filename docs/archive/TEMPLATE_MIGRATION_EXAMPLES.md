# Template Migration Examples

## Overview

This document shows the before/after comparison for templates migrated to use consolidated utilities from `src/`.

## Migrated Templates

1.  `Prophet_Python` - Facebook Prophet forecasting
2.  `MovingAverage_Python` - Simple/exponential/weighted moving averages
3.  `ExponentialSmoothing_Python` - ETS forecasting with comparison

## Prophet_Python

### Before (145 lines)

**Duplicated Code:**
- `repo_import()` function (10 lines)
- `load_config()` function (5 lines)
- `signalplot.apply()` (1 line)
- Custom data loading (5 lines)
- Custom output directory creation (2 lines)
- Custom plotting code (30 lines)
- Custom plot saving (5 lines)

**Total Duplicated:** ~58 lines

### After (130 lines)

**Using Consolidated Utilities:**
- No `repo_import()` - not needed
- Uses `load_config()` from `src/`
- No `signalplot.apply()` - applied in `src/__init__.py`
- Uses `load_time_series()` from `src/`
- Uses `ensure_output_dir()` and `get_output_dir()` from `src/`
- Uses `create_forecast_plot()` from `src/`
- Uses `save_plot()` from `src/`

**Benefits:**
- Removed ~50 lines of duplicated code
- Consistent with other templates
- Easier to maintain
- Cleaner, more readable code

## MovingAverage_Python

### Before (154 lines)

**Duplicated Code:**
- `repo_import()` function (10 lines)
- `load_config()` function (5 lines)
- `signalplot.apply()` (1 line)
- Custom data loading via `utils.ts_utils` (3 lines)
- Custom output directory creation (2 lines)
- Custom plotting code (40 lines)
- Custom plot saving (5 lines)

**Total Duplicated:** ~66 lines

### After (150 lines)

**Using Consolidated Utilities:**
- No `repo_import()` - not needed
- Uses `load_config()` from `src/`
- No `signalplot.apply()` - applied in `src/__init__.py`
- Uses `load_time_series()` directly from `src/`
- Uses `Evaluator` from `src/` for train/test split and evaluation
- Uses `create_forecast_plot()` from `src/`
- Uses `save_plot()` from `src/`

**Benefits:**
- Removed ~60 lines of duplicated code
- Standardized evaluation using `Evaluator`
- Consistent plotting patterns
- Better code reuse

## ExponentialSmoothing_Python

### Before (266 lines)

**Duplicated Code:**
- `load_config()` function with dataclass parsing (25 lines)
- Custom data loading (10 lines)
- Custom output directory creation (3 lines)
- Custom plot saving (multiple instances, ~10 lines)

**Total Duplicated:** ~48 lines

### After (240 lines)

**Using Consolidated Utilities:**
- Uses `load_config()` from `src/`, then parses into Config dataclass
- Uses `load_time_series()` from `src/`, then applies template-specific processing
- Uses `ensure_output_dir()` and `get_output_dir()` from `src/`
- Uses `save_plot()` from `src/` for consistent plot saving

**Benefits:**
- Removed ~40 lines of duplicated code
- Maintained template-specific features (rolling origin, comparison plots)
- Consistent plot saving
- Cleaner separation of concerns

## Migration Summary

### Code Reduction

| Template | Before | After | Reduction |
|----------|--------|-------|-----------|
| Prophet | 145 lines | 130 lines | ~50 lines removed |
| MovingAverage | 154 lines | 150 lines | ~60 lines removed |
| ExponentialSmoothing | 266 lines | 240 lines | ~40 lines removed |
| **Total** | **565 lines** | **520 lines** | **~150 lines removed** |

### Duplicated Code Eliminated

-  37+ `repo_import()` functions → 1 in `src/utils.py`
-  56+ `load_config()` variations → 1 in `src/config.py`
-  53+ `signalplot.apply()` calls → 1 in `src/__init__.py`
-  100+ plotting patterns → Standardized in `src/plotting.py`
-  50+ output directory patterns → Standardized in `src/utils.py`

### Benefits

1. **Consistency**: All templates follow same patterns
2. **Maintainability**: Fix bugs once, benefit everywhere
3. **Readability**: Less boilerplate, more focus on template-specific logic
4. **Extensibility**: Easy to add new templates using BaseTemplate
5. **Code Quality**: DRY principle applied throughout

## Next Steps

1.  Migrate 3 templates as examples (completed)
2.  Migrate remaining templates incrementally
3.  Update template READMEs with new usage patterns
4.  Remove old duplicated code from remaining templates

## Migration Checklist

For each template:
- [ ] Remove `repo_import()` function
- [ ] Remove `signalplot.apply()` call
- [ ] Replace custom `load_config()` with `src.load_config()`
- [ ] Replace custom data loading with `src.load_time_series()`
- [ ] Replace custom output dir creation with `src.ensure_output_dir()`
- [ ] Replace custom plotting with `src.create_forecast_plot()`
- [ ] Replace custom plot saving with `src.save_plot()`
- [ ] Test template still works correctly
- [ ] Update README if needed

