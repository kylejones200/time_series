# STUMPY + PyOD for Anomaly Detection

Matrix profile (STUMPY) and Python Outlier Detection (PyOD) for time series anomaly detection.

## Features

- ✅ STUMPY matrix profile for pattern-based anomaly detection
- ✅ PyOD with multiple algorithms (Isolation Forest, LOF, OCSVM)
- ✅ Comprehensive evaluation metrics (precision, recall, F1)
- ✅ Comparative visualization of both methods

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file in the shared `data/` directory
2. Update `config.yaml` with your data file name and column names
3. Run:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- **use_stumpy**: Enable/disable STUMPY detection
- **stumpy_window**: Window size for matrix profile calculation
- **stumpy_percentile**: Percentile threshold for anomalies
- **use_pyod**: Enable/disable PyOD detection
- **pyod_method**: Algorithm to use (`IForest`, `LOF`, `OCSVM`)
- **pyod_contamination**: Expected proportion of anomalies

## Methods

### STUMPY
- Matrix profile for pattern discovery
- Detects unusual subsequences
- Window-based approach

### PyOD
- **IForest**: Isolation Forest for anomaly detection
- **LOF**: Local Outlier Factor
- **OCSVM**: One-Class SVM

## Outputs

- `outputs/stumpy_pyod_anomalies.png`: Anomaly detection visualization with both methods
- Console output: Evaluation metrics for each method

## Notes

- STUMPY is best for pattern-based anomalies
- PyOD is best for point anomalies
- Both methods can be used together for comprehensive detection
- Optional true anomaly labels for evaluation if available

