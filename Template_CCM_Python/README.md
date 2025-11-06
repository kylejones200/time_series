# Convergent Cross Mapping (CCM)

Causal inference method for detecting causality in time series using state space reconstruction and cross-mapping.

## Features

- ✅ Time-delay embedding for state space reconstruction
- ✅ Cross-mapping between time series
- ✅ Bidirectional causality detection
- ✅ Correlation-based causality strength measurement
- ✅ Four-panel visualization (time series, scatter plots)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your two time series data files in the shared `data/` directory
2. Update `config.yaml` with your data file names and column names
3. Run:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- **data**: 
  - `series1_file` and `series2_file`: Input file names
  - `series1_col` and `series2_col`: Column names for values
  - `series1_name` and `series2_name`: Display names
- **model**:
  - `delay`: Time delay for embedding (typically 1-20)
  - `dimension`: Embedding dimension (typically 2-5)
  - `n_neighbors`: Number of neighbors (default: dimension + 1)
  - `normalize`: Whether to normalize series before analysis
  - `causality_threshold`: Threshold for bidirectional detection

## Method

CCM detects causality by:
1. Reconstructing state space manifolds using time-delay embedding
2. Cross-mapping: using one series' manifold to predict the other
3. Measuring correlation between actual and predicted values
4. Higher correlation indicates stronger causal relationship

## Outputs

- `outputs/ccm_analysis.png`: Four-panel plot showing:
  - Time series with cross-mapped predictions (both directions)
  - Scatter plots of actual vs predicted values

## Interpretation

- **High correlation (→)**: Strong causal relationship in that direction
- **Bidirectional**: Both series influence each other
- **Asymmetric**: One direction stronger than the other (unidirectional causality)

## Notes

- Best for nonlinear, potentially chaotic systems
- Requires sufficient data length (typically >100 points)
- Embedding parameters (delay, dimension) may need tuning
- Normalization recommended for series with different scales

