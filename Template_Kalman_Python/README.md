# Kalman Filters: State Space Models

Kalman filtering and smoothing for time series analysis using state space models.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: State space model parameters (dimensions, noise, matrices)
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Model Parameters

- **state_dimension**: Dimension of state vector
- **measurement_dimension**: Dimension of measurement vector
- **state_transition**: State transition matrix (F)
- **measurement_matrix**: Measurement matrix (H)
- **measurement_noise**: Measurement noise covariance (R)
- **process_noise**: Process noise variance (Q)

## Outputs

Filtered estimates and plots saved to `outputs/` directory.

