#!/usr/bin/env python3
"""
Intelligent model selection based on data characteristics.

Analyzes your data and recommends the best forecasting methods.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_time_series


def analyze_series(series: pd.Series) -> Dict:
    """Analyze time series characteristics."""
    n = len(series)
    
    # Basic stats
    mean_val = series.mean()
    std_val = series.std()
    cv = std_val / mean_val if mean_val != 0 else np.inf  # Coefficient of variation
    
    # Trend detection
    first_quarter = series.iloc[:n//4].mean()
    last_quarter = series.iloc[-n//4:].mean()
    trend_strength = abs(last_quarter - first_quarter) / mean_val if mean_val != 0 else 0
    
    # Seasonality detection (simple autocorrelation)
    has_seasonality = False
    seasonality_strength = 0
    if n > 24:
        # Check multiple lags
        for lag in [4, 7, 12, 24]:
            if lag < n:
                autocorr = series.autocorr(lag=lag)
                if abs(autocorr) > 0.3:
                    has_seasonality = True
                    seasonality_strength = max(seasonality_strength, abs(autocorr))
    
    # Stationarity (simple check)
    is_stationary = cv < 0.5 and trend_strength < 0.2
    
    # Volatility
    returns = series.pct_change().dropna()
    volatility = returns.std() if len(returns) > 0 else 0
    
    return {
        "length": n,
        "mean": mean_val,
        "std": std_val,
        "cv": cv,
        "trend_strength": trend_strength,
        "has_trend": trend_strength > 0.1,
        "has_seasonality": has_seasonality,
        "seasonality_strength": seasonality_strength,
        "is_stationary": is_stationary,
        "volatility": volatility,
    }


def recommend_models(characteristics: Dict) -> List[Tuple[str, str, float]]:
    """
    Recommend models based on characteristics.
    
    Returns:
    --------
    List of (template_name, reason, confidence) tuples
    """
    recommendations = []
    n = characteristics["length"]
    
    # Short series (< 30 points)
    if n < 30:
        recommendations.extend([
            ("MovingAverage_Python", "Simple baseline for short series", 0.9),
            ("ExponentialSmoothing_Python", "Good for short series with trends", 0.8),
        ])
    
    # Medium series (30-100 points)
    elif n < 100:
        recommendations.extend([
            ("ARIMA_Python", "Classical method, interpretable", 0.9),
            ("ExponentialSmoothing_Python", "Handles trends and seasonality", 0.85),
        ])
        
        if characteristics["has_seasonality"]:
            recommendations.append(("Prophet_Python", "Excellent for seasonal data", 0.95))
        else:
            recommendations.append(("ARIMA_Python", "Good for non-seasonal data", 0.9))
    
    # Long series (100+ points)
    else:
        # Always recommend these for long series
        recommendations.extend([
            ("ARIMA_Python", "Classical method, interpretable", 0.85),
            ("StatsForecast_Python", "Fast and accurate", 0.9),
        ])
        
        if characteristics["has_seasonality"]:
            recommendations.extend([
                ("Prophet_Python", "Best for strong seasonality", 0.95),
                ("Darts_Python", "Multiple methods, handles seasonality", 0.85),
            ])
        
        if n > 200:
            recommendations.extend([
                ("LSTM_Python", "Deep learning for complex patterns", 0.8),
                ("Chronos_Python", "Foundation model, pre-trained", 0.75),
            ])
    
    # Special cases
    if characteristics["volatility"] > 0.1:
        recommendations.append(("Volatility_Python", "High volatility detected", 0.7))
    
    if not characteristics["is_stationary"]:
        recommendations.append(("Differencing_Python", "Non-stationary data", 0.8))
    
    # Sort by confidence
    recommendations.sort(key=lambda x: x[2], reverse=True)
    
    return recommendations


def print_recommendations(data_path: str):
    """Analyze data and print recommendations."""
    print("\n" + "=" * 70)
    print("  Model Selection Recommendations")
    print("=" * 70 + "\n")
    
    print(f"Analyzing: {data_path}\n")
    
    try:
        series = load_time_series(data_path, date_column="date", value_column="value")
    except Exception as e:
        print(f" Error loading data: {e}")
        print("\nExpected format:")
        print("  date,value")
        print("  2020-01-01,100.5")
        return
    
    # Analyze
    print("Analyzing data characteristics...")
    characteristics = analyze_series(series)
    
    print("\n Data Characteristics:")
    print(f"  Length: {characteristics['length']} points")
    print(f"  Mean: {characteristics['mean']:.2f}")
    print(f"  Std Dev: {characteristics['std']:.2f}")
    print(f"  Coefficient of Variation: {characteristics['cv']:.2f}")
    print(f"  Trend: {'Yes' if characteristics['has_trend'] else 'No'} (strength: {characteristics['trend_strength']:.2f})")
    print(f"  Seasonality: {'Yes' if characteristics['has_seasonality'] else 'No'} (strength: {characteristics['seasonality_strength']:.2f})")
    print(f"  Stationary: {'Yes' if characteristics['is_stationary'] else 'No'}")
    print(f"  Volatility: {characteristics['volatility']:.4f}")
    
    # Get recommendations
    recommendations = recommend_models(characteristics)
    
    print("\n Recommended Models (sorted by confidence):")
    print("-" * 70)
    
    for i, (template, reason, confidence) in enumerate(recommendations[:10], 1):
        confidence_bar = "█" * int(confidence * 10)
        print(f"\n{i}. {template}")
        print(f"   {reason}")
        print(f"   Confidence: {confidence:.0%} {confidence_bar}")
    
    print("\n Next Steps:")
    print("  1. Run a single model:")
    print(f"     python forecast.py run {recommendations[0][0]} --data {data_path}")
    print("  2. Benchmark top recommendations:")
    print(f"     python forecast.py benchmark --data {data_path} --templates {' '.join([r[0] for r in recommendations[:5]])}")
    print("  3. Try the quick start wizard:")
    print("     python scripts/quick_start.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent model selection")
    parser.add_argument("data", help="Path to time series data")
    
    args = parser.parse_args()
    
    print_recommendations(args.data)

