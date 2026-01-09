#!/usr/bin/env python3
"""
Generate example production data for testing forecasting models.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_exponential_decline(qi=100.0, Di=0.05, periods=36, well_id="WELL_001"):
    """Generate exponential decline curve data."""
    dates = pd.date_range("2020-01-01", periods=periods, freq="MS")
    months = np.arange(periods)
    rates = qi * np.exp(-Di * months)

    df = pd.DataFrame(
        {
            "well_id": well_id,
            "date": dates,
            "oil_rate": rates,
            "gas_rate": rates * 0.5,
            "water_rate": rates * 0.1,
            "cum_oil": np.cumsum(rates) * 30,  # Approximate monthly production
            "cum_gas": np.cumsum(rates * 0.5) * 30,
        }
    )

    return df


def generate_hyperbolic_decline(qi=100.0, Di=0.05, b=0.5, periods=36, well_id="WELL_002"):
    """Generate hyperbolic decline curve data."""
    dates = pd.date_range("2020-01-01", periods=periods, freq="MS")
    months = np.arange(periods)
    rates = qi / (1 + b * Di * months) ** (1 / b)

    df = pd.DataFrame(
        {
            "well_id": well_id,
            "date": dates,
            "oil_rate": rates,
            "gas_rate": rates * 0.5,
            "water_rate": rates * 0.1,
            "cum_oil": np.cumsum(rates) * 30,
            "cum_gas": np.cumsum(rates * 0.5) * 30,
        }
    )

    return df


def main():
    """Generate example production data files."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # Generate single well data (exponential decline)
    df1 = generate_exponential_decline(qi=100.0, Di=0.05, periods=36, well_id="WELL_001")
    df1.to_csv(output_dir / "well_production.csv", index=False)
    print(f"Generated {output_dir / 'well_production.csv'} ({len(df1)} rows)")

    # Generate multi-well data
    df2 = generate_exponential_decline(qi=120.0, Di=0.04, periods=36, well_id="WELL_001")
    df3 = generate_hyperbolic_decline(qi=100.0, Di=0.05, b=0.6, periods=36, well_id="WELL_002")
    df4 = generate_exponential_decline(qi=80.0, Di=0.06, periods=36, well_id="WELL_003")

    multi_well = pd.concat([df2, df3, df4], ignore_index=True)
    multi_well.to_csv(output_dir / "multi_well.csv", index=False)
    print(f"Generated {output_dir / 'multi_well.csv'} ({len(multi_well)} rows)")

    # Generate synthetic decline with noise
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    months = np.arange(36)
    qi, Di = 100.0, 0.05
    rates = qi * np.exp(-Di * months)
    # Add some noise
    rates = rates * (1 + np.random.normal(0, 0.05, len(rates)))

    df_synthetic = pd.DataFrame(
        {
            "well_id": "WELL_SYNTHETIC",
            "date": dates,
            "oil_rate": np.maximum(rates, 1.0),  # Ensure positive
            "gas_rate": rates * 0.5,
            "water_rate": rates * 0.1,
            "cum_oil": np.cumsum(np.maximum(rates, 1.0)) * 30,
            "cum_gas": np.cumsum(rates * 0.5) * 30,
        }
    )

    df_synthetic.to_csv(output_dir / "synthetic_decline.csv", index=False)
    print(f"Generated {output_dir / 'synthetic_decline.csv'} ({len(df_synthetic)} rows)")

    print("\n Example production data generated successfully!")


if __name__ == "__main__":
    main()

