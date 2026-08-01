# CI Improvements: Lessons from Other Repos

This document captures improvements applied to prevent cascading CI errors based on lessons learned from other repositories.

## Issues Addressed

### 1. Optional Dependencies

**Problem**: Optional dependencies imported with `try/except` but used without defensive checks, causing failures when modules are unavailable.

**Solution Applied**:
- Changed generic `Exception` to `ImportError` for clearer error handling
- Added explicit `None` checks before using optional modules
- Added informative warning messages when optional dependencies are unavailable

**Files Changed**:
- `STUMPY_PyOD_Python/main.py`:
  - Changed `except Exception` to `except ImportError`
  - Added defensive checks: `if stumpy is None:` before usage
  - Added defensive checks: `if IForest is None or LOF is None or OCSVM is None:` before usage

### 2. File Handling on Windows

**Problem**: File handles not properly closed before deletion, causing locking issues on Windows.

**Solution Applied**:
- Explicitly close figures after saving (already had `plt.close(fig)` but improved)
- Use explicit UTF-8 encoding for cross-platform compatibility
- Ensure file handles are properly closed in context managers
- Added `fig.canvas.flush_events()` to ensure data is written before closing

**Files Changed**:
- `src/plotting.py`:
  - Wrapped `fig.savefig()` in try/finally to ensure closing
  - Added `fig.canvas.flush_events()` before closing
  - Explicitly close figure with `plt.close(fig)` in finally block
  
- `src/config.py`:
  - Added explicit `encoding="utf-8"` to file open
  - File already properly closed via context manager
  
- `src/base_template.py`:
  - Added explicit `encoding="utf-8"` to `df.to_csv()`
  
- `src/loader.py`:
  - Added explicit `encoding="utf-8"` to `pd.read_csv()`
  - Added error handling for invalid dates
  - Added file existence check before reading

### 3. Type Hints for Optional Dependencies

**Problem**: Type hints referencing optional dependencies directly can cause import errors during type checking.

**Solution Applied**:
- No direct type hints used for optional modules in STUMPY_PyOD template
- If needed in future, use string forward references: `-> "OptionalModule.Type"` instead of `-> OptionalModule.Type`

**Note**: Currently, optional dependencies in STUMPY_PyOD are not used in type hints, so this is not an immediate issue. However, if type hints are added in the future, they should use string forward references.

## Prevention Guidelines

Going forward, follow these patterns:

### 1. Optional Dependencies Pattern

```python
try:
    import optional_module
    from optional_module import OptionalClass
except ImportError:  # Not Exception - be specific
    optional_module = None  # type: ignore
    OptionalClass = None  # type: ignore

def use_optional_thing():
    if OptionalClass is None:
        print("Warning: optional_module not available. Skipping.")
        return
    # Use OptionalClass safely
```

### 2. File Handling Pattern

```python
# Always use explicit encoding and ensure proper closing
with open(file_path, encoding="utf-8") as f:
    data = f.read()

# For matplotlib figures
try:
    fig.savefig(output_path, dpi=300)
    fig.canvas.flush_events()
finally:
    plt.close(fig)

# For pandas CSV writing
df.to_csv(output_path, encoding="utf-8")
```

### 3. Type Hints for Optional Dependencies

```python
# If using type hints for optional dependencies, use string forward references
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from optional_module import OptionalClass

def process(data) -> "Optional[OptionalClass]":
    # Implementation
    pass
```

## Testing Checklist

Before committing, verify:

- [ ] Optional dependencies fail gracefully with clear messages
- [ ] File operations work on Windows (test if possible)
- [ ] No import errors during type checking (run `mypy` or similar)
- [ ] All file handles are properly closed
- [ ] UTF-8 encoding is explicitly specified for file operations

## Status

 All known issues addressed
 Patterns documented for future development
 CI improvements applied across codebase

