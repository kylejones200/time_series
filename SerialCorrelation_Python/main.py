#!/usr/bin/env python3
"""
Serial Correlation Analysis for Time Series
Tests and corrections for serial correlation in time series regression models.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
from statsmodels.regression.linear_model import GLS, GLSAR
from statsmodels.graphics.tsaplots import plot_acf


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data using consolidated loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value")
    )
    
    # Convert to DataFrame for feature engineering
    df = pd.DataFrame({config["data"]["value_col"]: series})
    
    y_col = config["model"]["y_col"]
    x_cols = config["model"]["x_cols"]
    
    # Create lags if configured
    if config["model"].get("create_lags", False):
        from utils.ts_utils import create_lags
        for lag in range(1, config["model"].get("max_lags", 2) + 1):
            df = create_lags(df, x_cols[0], [lag])
            x_cols.append(f"{x_cols[0]}_lag{lag}")
    
    df = df[[y_col] + x_cols].dropna()
    
    y = df[y_col]
    X = df[x_cols]
    X = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    print("\nOLS Regression Results:")
    print("=" * 70)
    print(model.summary())
    
    residuals = model.resid
    
    # Breusch-Godfrey test
    bg_test = acorr_breusch_godfrey(model, nlags=config["model"]["test_lags"])
    print(f"\nBreusch-Godfrey Test p-value: {bg_test[1]:.4f}")
    if bg_test[1] < 0.05:
        print("→ Serial correlation detected (p < 0.05)")
    else:
        print("→ No significant serial correlation (p >= 0.05)")
    
    # Ljung-Box test
    lb_test = acorr_ljungbox(
        residuals, lags=config["model"]["test_lags"], return_df=True
    )
    print(f"\nLjung-Box Test:")
    print(lb_test)
    
    # Apply corrections if configured
    if config["model"].get("apply_corrections", False):
        print("\n" + "=" * 70)
        print("Corrected Models:")
        print("=" * 70)
        
        gls_model = GLS(y, X).fit()
        print("\nGLS Model:")
        print(gls_model.summary())
        
        cochrane_orcutt = GLSAR(y, X, rho=1).iterative_fit(maxiter=100)
        print("\nCochrane-Orcutt Model:")
        print(cochrane_orcutt.summary())
    
    # Create visualization
    fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))
    
    plot_acf(residuals, lags=config["model"]["test_lags"], ax=ax)
    ax.set_title("Residual Autocorrelation Function (ACF)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    save_plot(fig, output_dir / "serial_correlation_acf.png", dpi=300)
    print(f"\nPlot saved to: {output_dir / 'serial_correlation_acf.png'}")
    
    print("\n Serial correlation analysis complete")
    
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
