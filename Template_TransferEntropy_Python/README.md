# Transfer Entropy for Causal Inference

Information-theoretic causal inference using transfer entropy to detect information flow between time series.

## Features

- ✅ Transfer entropy computation
- ✅ Bidirectional causality detection
- ✅ Rolling transfer entropy for time-varying relationships
- ✅ Information-theoretic approach (no linearity assumptions)
- ✅ Handles nonlinear dependencies

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
  - `difference`: Whether to difference series before analysis
- **model**:
  - `k`: Lag order for transfer entropy
  - `bins`: Number of bins for discretization
  - `compute_rolling`: Whether to compute rolling transfer entropy
  - `rolling_window`: Window size for rolling computation

## Method

Transfer Entropy measures:
- Information flow from one series to another
- Asymmetric causality (direction matters)
- Nonlinear dependencies
- Based on information theory (Shannon entropy)

## Outputs

- `outputs/transfer_entropy_analysis.png`: Four-panel plot showing:
  - Time series
  - Rolling transfer entropy
  - Static transfer entropy comparison
  - Bidirectional scatter plot

## Interpretation

- **Higher TE value**: Stronger information flow in that direction
- **Asymmetric**: One direction stronger than the other
- **Bidirectional**: Similar TE in both directions
- **Rolling TE**: Shows how causality changes over time

## Notes

- Requires discretization (bins parameter)
- Best for nonlinear, potentially chaotic systems
- More robust than linear methods (Granger causality)
- Computationally intensive for rolling windows
- Typically requires differencing for non-stationary data

