#!/usr/bin/env python3
"""Convert EIA net generation monthly data into ML-ready format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def convert_net_generation_dataset() -> Path:
    data_dir = Path(__file__).resolve().parent.parent
    raw_path = data_dir / "Net_generation_United_States_all_sectors_monthly.csv"
    output_path = raw_path.with_name(f"{raw_path.stem}_ml-ready.csv")

    df = pd.read_csv(raw_path, skiprows=4)
    df.columns = [col.strip() for col in df.columns]

    df["Month"] = pd.to_datetime(df["Month"], format="%b %Y")
    df["Month"] = df["Month"].dt.strftime("%Y-%m-%d")

    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    output_path = convert_net_generation_dataset()
    print(f"✓ Net generation dataset converted: {output_path.name}")


if __name__ == "__main__":
    main()
