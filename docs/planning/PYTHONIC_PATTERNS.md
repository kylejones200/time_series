# Pythonic Patterns (No if/elif/else)

## Dictionary Lookups Instead of Conditionals

**Before:**
```python
if condition:
    value = option_a
else:
    value = option_b
```

**After:**
```python
value_map = {True: option_a, False: option_b}
value = value_map[condition]
```

## List Comprehensions for Conditional Side Effects

**Before:**
```python
if config['preprocessing']['wavelet_denoise']:
    for f in features:
        df_train[f] = wavelet_denoise(...)
```

**After:**
```python
[df_train.__setitem__(f, wavelet_denoise(...)) 
 for f in features 
 for _ in [None] if config['preprocessing']['wavelet_denoise']]
```

## Dictionary-Based Function Dispatch

**Before:**
```python
if nrows == 1 and ncols == 1:
    axes = [axes]
elif nrows == 1 or ncols == 1:
    axes = axes.flatten()
else:
    axes = axes.flatten()
```

**After:**
```python
axes_map = {
    (True, True): lambda ax: [ax],
    (True, False): lambda ax: ax.flatten(),
    (False, True): lambda ax: ax.flatten(),
    (False, False): lambda ax: ax.flatten(),
}
key = (nrows == 1, ncols == 1)
axes = axes_map[key](axes)
```

## Tuple Unpacking from Maps

**Before:**
```python
if config['preprocessing']['standardize']:
    scaled_data = StandardScaler().fit_transform(df_train)
    scaler = StandardScaler()
else:
    scaled_data = df_train.values
    scaler = None
```

**After:**
```python
scaler_map = {
    True: (StandardScaler().fit_transform(df_train), StandardScaler()),
    False: (df_train.values, None)
}
scaled_data, scaler = scaler_map[config['preprocessing']['standardize']]
```

## Config-Driven Styling

All matplotlib styling is controlled via `config.yaml`:

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
```

## Benefits

1. **No if/elif/else chains** - Cleaner, more declarative code
2. **Data-driven** - Logic in config, not code
3. **Functional** - Side effects via comprehensions
4. **Pythonic** - Uses Python idioms effectively
5. **Maintainable** - Easy to extend with new options

