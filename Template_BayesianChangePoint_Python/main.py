#!/usr/bin/env python3
"""
Bayesian Change Point Detection
Bayesian MCMC approach to detect change points in time series using PyMC.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import util
import pymc as pm
import arviz as az


def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj


plotting_utils = repo_import("utils.plotting_utils")
setup_figure = plotting_utils.setup_figure
apply_legend = plotting_utils.apply_legend
save_plot = plotting_utils.save_plot
apply_plot_style = plotting_utils.apply_plot_style

ts_utils = repo_import("utils.ts_utils")
load_ts_data = ts_utils.load_ts_data
ensure_datetime_index = ts_utils.ensure_datetime_index


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_change_point_bayesian(data, config):
    """Detect change point using Bayesian MCMC."""
    n = len(data)
    
    with pm.Model() as model:
        tau = pm.DiscreteUniform("tau", lower=0, upper=n-1)
        
        lambda_1 = pm.Exponential("lambda_1", lam=config['model']['lambda_prior'])
        lambda_2 = pm.Exponential("lambda_2", lam=config['model']['lambda_prior'])
        
        idx = np.arange(n)
        lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)
        
        observation = pm.Poisson("obs", mu=lambda_, observed=data.values)
        
        trace = pm.sample(
            config['model']['draws'],
            tune=config['model']['tune'],
            chains=config['model']['chains'],
            cores=config['model']['cores'],
            return_inferencedata=True,
            random_seed=config['model']['random_seed']
        )
    
    return trace, model


def detect_change_point_frequentist(data, config):
    """Detect change point using frequentist method (for comparison)."""
    from scipy import stats
    
    n = len(data)
    best_tau = 0
    best_p_value = 1.0
    
    for tau in range(1, n-1):
        before = data.iloc[:tau]
        after = data.iloc[tau:]
        
        if len(before) > 1 and len(after) > 1:
            stat, p_value = stats.ks_2samp(before, after)
            if p_value < best_p_value:
                best_p_value = p_value
                best_tau = tau
    
    return best_tau, best_p_value


def main():
    config = load_config()
    
    df = load_ts_data(
        data_path=Path(__file__).parent.parent / 'data' / config['data']['input_file'],
        date_col=config['data']['date_col'],
        value_col=config['data']['value_col']
    )
    df = ensure_datetime_index(df, time_col=config['data']['date_col'])
    
    if config['model']['method'] == 'bayesian':
        trace, model = detect_change_point_bayesian(df[config['data']['value_col']], config)
        
        burn_in = config['model']['burn_in']
        tau_samples = trace.posterior['tau'].values.flatten()[burn_in:]
        lambda_1_samples = trace.posterior['lambda_1'].values.flatten()[burn_in:]
        lambda_2_samples = trace.posterior['lambda_2'].values.flatten()[burn_in:]
        
        tau_mean = int(np.mean(tau_samples))
        lambda_1_mean = np.mean(lambda_1_samples)
        lambda_2_mean = np.mean(lambda_2_samples)
        
        print("\nBayesian Change Point Detection Results:")
        print("=" * 70)
        print(f"Estimated change point (mean): {tau_mean}")
        print(f"Change point date: {df.index[tau_mean]}")
        print(f"Lambda before change: {lambda_1_mean:.4f}")
        print(f"Lambda after change: {lambda_2_mean:.4f}")
        print(f"95% Credible Interval for tau: [{int(np.percentile(tau_samples, 2.5))}, {int(np.percentile(tau_samples, 97.5))}]")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for ax in axes.flatten():
            apply_plot_style(ax, {'plotting': config['plotting']})
        
        axes[0, 0].plot(df.index, df[config['data']['value_col']].values,
                        'k-', linewidth=config['plotting']['linewidth'],
                        alpha=config['plotting']['alpha'])
        axes[0, 0].axvline(x=df.index[tau_mean], color='r', linestyle='--',
                           linewidth=config['plotting']['linewidth'],
                           label=f'Change Point (τ={tau_mean})')
        axes[0, 0].set_title('Time Series with Change Point')
        axes[0, 0].set_ylabel('Value')
        apply_legend(axes[0, 0], config['plotting']['legend'])
        
        axes[0, 1].hist(tau_samples, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=tau_mean, color='r', linestyle='--',
                           linewidth=config['plotting']['linewidth'])
        axes[0, 1].set_title('Posterior Distribution of Change Point (τ)')
        axes[0, 1].set_xlabel('Change Point Index')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].hist(lambda_1_samples, bins=50, edgecolor='black', alpha=0.7, label='λ₁')
        axes[1, 0].hist(lambda_2_samples, bins=50, edgecolor='black', alpha=0.7, label='λ₂')
        axes[1, 0].set_title('Posterior Distributions of λ₁ and λ₂')
        axes[1, 0].set_xlabel('Lambda')
        axes[1, 0].set_ylabel('Frequency')
        apply_legend(axes[1, 0], config['plotting']['legend'])
        
        axes[1, 1].plot(tau_samples[:500], 'k-', linewidth=1, alpha=0.7)
        axes[1, 1].set_title('MCMC Trace Plot (τ)')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Change Point Index')
        
        plt.tight_layout()
        
        output_path = Path(__file__).parent / "outputs" / "bayesian_change_point.png"
        save_plot(fig, output_path)
        plt.show()
        
    else:
        tau, p_value = detect_change_point_frequentist(df[config['data']['value_col']], config)
        print(f"\nFrequentist Change Point Detection:")
        print(f"Change point: {tau}")
        print(f"Change point date: {df.index[tau]}")
        print(f"p-value: {p_value:.4f}")
        
        fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
        apply_plot_style(ax, {'plotting': config['plotting']})
        
        ax.plot(df.index, df[config['data']['value_col']].values,
                'k-', linewidth=config['plotting']['linewidth'],
                alpha=config['plotting']['alpha'])
        ax.axvline(x=df.index[tau], color='r', linestyle='--',
                   linewidth=config['plotting']['linewidth'],
                   label=f'Change Point (τ={tau})')
        
        ax.set_title(config['plot_titles']['change_point'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        apply_legend(ax, config['plotting']['legend'])
        
        output_path = Path(__file__).parent / "outputs" / "change_point.png"
        save_plot(fig, output_path)
        plt.show()


if __name__ == "__main__":
    main()

