# Proposed Project Structure

Instead of notebooks, each time series project would have:

```
project_name/
├── config.yaml          # Parameters, data paths, model settings
├── requirements.txt     # Python dependencies
├── main.py             # Main execution script
├── data/               # Input data files
│   └── *.csv
└── outputs/            # Generated plots and results
    └── *.png
```

## Example Structure

### config.yaml
```yaml
data:
  input_file: "data/turbine_telemetry.csv"
  timestamp_col: "timestamp"
  target_col: "voltage"
  features:
    - "gearbox_vibration"
    - "generator_vibration"
    - "temperature"

model:
  type: "isolation_forest"
  contamination: 0.01
  n_estimators: 100

preprocessing:
  wavelet_denoise: true
  wavelet_type: "db6"
  standardize: true

output:
  plot_format: "png"
  save_plots: true
  output_dir: "outputs"
```

### main.py
```python
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# ... imports

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Load data
    df = pd.read_csv(config['data']['input_file'])
    
    # Preprocess
    # ... preprocessing code
    
    # Train model
    # ... model code
    
    # Generate outputs
    # ... visualization code
    
    print("✓ Complete")

if __name__ == "__main__":
    main()
```

## Benefits

1. **Version Control**: Easy to track changes in code and config separately
2. **Reproducibility**: Config files ensure consistent runs
3. **Modularity**: Can import functions from other projects
4. **Testing**: Can write unit tests for functions
5. **CI/CD**: Can run in automated pipelines
6. **Professional**: Industry-standard structure

