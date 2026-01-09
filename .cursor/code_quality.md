# Code Quality Guidelines

This document outlines the code quality patterns and lessons learned to prevent CI issues and cascading errors.

## Optional Dependencies Pattern

### Use String Forward References

** DO:**
```python
def _create_model(self) -> "Sequential":
    """Create model."""
    if not KERAS_AVAILABLE or Sequential is None:
        raise ImportError("Keras/TensorFlow is required.")
    return Sequential(...)
```

** DON'T:**
```python
def _create_model(self) -> Sequential:  # Causes ImportError if Sequential not available
    """Create model."""
    return Sequential(...)
```

### Import Pattern

** DO:**
```python
# Import only in try/except, use string forward refs in type hints
# Use ImportError (not generic Exception) for clearer error handling
try:
    import stumpy
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
except ImportError:  # Not Exception - be specific
    stumpy = None  # type: ignore[assignment,misc]
    IForest = None  # type: ignore[assignment,misc]
    LOF = None  # type: ignore[assignment,misc]
    OCSVM = None  # type: ignore[assignment,misc]

def use_optional_module():
    if stumpy is None:
        print("Warning: stumpy not available. Skipping.")
        return
    # Use stumpy safely
    result = stumpy.stump(data, m=window)
```

** DON'T:**
```python
# Using generic Exception catches too much
try:
    import optional_module
except Exception:  # Too broad - catches syntax errors, etc.
    optional_module = None

# Missing defensive checks
def use_optional_module():
    # No check - can fail if import failed
    result = optional_module.function()  # AttributeError if None
```

## Defensive Checks

### Initialize Attributes

** DO:**
```python
def __init__(self):
    super().__init__()
    self.model_ = None
    self.device_ = None
    self.reconstruction_errors_ = None
    # Initialize all attributes to avoid AttributeError
```

### Check Before Use

** DO:**
```python
def score_samples(self, X):
    if self.model_ is None:
        raise ValueError("Detector must be fitted before scoring.")
    if not TORCH_AVAILABLE or torch is None:
        raise ImportError("PyTorch is required.")
    if self.device_ is None:
        raise RuntimeError("Device not initialized.")
    # Now safe to use
    self.model_.eval()
```

** DON'T:**
```python
def score_samples(self, X):
    # No checks - can cause AttributeError or TypeError
    self.model_.eval()
    X_tensor = torch.from_numpy(X)  # torch might be None
```

### Check Optional Dependencies

** DO:**
```python
if not KERAS_AVAILABLE or Sequential is None or EarlyStopping is None:
    raise ImportError("Keras/TensorFlow is required.")
```

** DON'T:**
```python
# Assuming Sequential is always available
model = Sequential(...)  # Can fail if import failed
```

## Base Class Pattern

** DO:**
```python
class StatisticalDetector(BaseDetector):
    def __init__(self, random_state: Optional[int] = None):
        super().__init__(random_state)
        self.threshold: Optional[float] = None  # Initialize attribute
    
    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Detector must be fitted before prediction.")
        scores = self.score_samples(X)
        return np.where(scores > self.threshold, -1, 1)
```

** DON'T:**
```python
class StatisticalDetector(BaseDetector):
    # No __init__ - threshold not initialized
    
    def predict(self, X):
        scores = self.score_samples(X)
        return np.where(scores > self.threshold, -1, 1)  # AttributeError if not set
```

## Platform-Specific Considerations

### File Handling (Windows Compatibility)

** DO:**
```python
# Always use explicit UTF-8 encoding for cross-platform compatibility
with open(filename, encoding="utf-8") as f:
    data = f.read()
# File automatically closed via context manager
os.remove(filename)  # Safe on all platforms

# For pandas CSV operations
df.to_csv(output_path, encoding="utf-8")

# For pandas CSV reading
df = pd.read_csv(file_path, encoding="utf-8")
```

** DON'T:**
```python
# Missing encoding - can cause issues on Windows
with open(filename, 'r') as f:
    data = f.read()

# Missing encoding in pandas
df.to_csv(output_path)  # May use system default encoding
df = pd.read_csv(file_path)  # May fail with non-UTF-8 files
```

### Matplotlib Figure Handling

** DO:**
```python
# Save and properly close figures to avoid file locking on Windows
def save_plot(fig, output_path, close=True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    
    # Flush to ensure data is written before closing (important on Windows)
    if fig.canvas is not None:
        fig.canvas.flush_events()
    
    # Close figure to avoid file locking issues on Windows
    if close:
        plt.close(fig)
    
    return output_path
```

** DON'T:**
```python
# Not flushing or closing can cause file locking on Windows
fig.savefig(output_path)
# File handle may remain open, causing issues on Windows
```

## Summary

1. **String Forward References**: Use `-> "Type"` for optional dependency types
2. **Defensive Checks**: Always check `is None` before using optional dependencies
3. **Initialize Attributes**: Set all attributes to `None` or default values in `__init__`
4. **Clear Error Messages**: Provide helpful error messages when dependencies missing
5. **Test Without Optional Deps**: Ensure core functionality works without optional packages
6. **File Encoding**: Always use explicit `encoding="utf-8"` for file operations
7. **Figure Handling**: Flush and close matplotlib figures to avoid Windows file locking
8. **ImportError vs Exception**: Use `except ImportError` (not `except Exception`) for optional dependencies

## Related Documentation

- See `docs/planning/CI_IMPROVEMENTS.md` for specific improvements applied to this repository
- See `docs/planning/DRY_CONSOLIDATION_PLAN.md` for code consolidation patterns

These patterns prevent cascading CI errors and make the codebase more robust across platforms.

