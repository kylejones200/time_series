# Bayesian Change Point Detection

Bayesian MCMC approach to detect change points in time series using PyMC. Converted from R implementation.

## Features

- ✅ Bayesian change point detection using MCMC
- ✅ PyMC implementation with Poisson likelihood
- ✅ Posterior distributions for change point location
- ✅ Credible intervals
- ✅ MCMC diagnostics and trace plots
- ✅ Frequentist method for comparison

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

- **method**: `bayesian` or `frequentist`
- **lambda_prior**: Prior parameter for lambda (rate parameters)
- **draws**: Number of MCMC samples
- **tune**: Number of tuning samples
- **chains**: Number of MCMC chains
- **burn_in**: Burn-in samples to discard

## Model

The Bayesian model assumes:
- Data follows a Poisson distribution
- Change point τ divides the series into two segments
- Each segment has its own rate parameter (λ₁, λ₂)
- Exponential priors on rate parameters

## Outputs

- `outputs/bayesian_change_point.png`: Four-panel plot showing:
  - Time series with detected change point
  - Posterior distribution of change point
  - Posterior distributions of λ₁ and λ₂
  - MCMC trace plots

## Notes

- Best for count data (Poisson likelihood)
- Can be adapted for other distributions (Normal, etc.)
- Bayesian approach provides uncertainty quantification
- Frequentist method uses Kolmogorov-Smirnov test for comparison

