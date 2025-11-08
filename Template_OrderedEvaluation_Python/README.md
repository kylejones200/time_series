# Ordered Model Evaluation

Simulate ordinal predictions for two competing models and evaluate them using
quadratic-weighted Cohen’s kappa, Wilcoxon tests, calibration plots, confusion
matrices, and simple policy-cost analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The defaults generate synthetic data. Configure parameters in `config.yaml`,
then run:

```bash
python main.py
```

Outputs (`outputs/`):

- `calibration.png` — predicted vs. true class frequencies
- `confusion_matrices.png` — confusion matrices for model A & B
- `policy_costs.csv` — expected cost based on simple intervention policy

Console output prints kappa and Wilcoxon statistics.
