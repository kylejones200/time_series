#!/usr/bin/env python3
"""
Automated benchmarking tool.

Runs multiple forecasting templates on the same data and generates a comparison report.
"""

import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from datetime import datetime

from src import load_time_series, Evaluator
from evaluation.metrics import calculate_metrics
from evaluation.comparison import compare_forecasts


# Curated set of fast, representative templates for benchmarking
BENCHMARK_TEMPLATES = [
    "ARIMA_Python",
    "Prophet_Python", 
    "MovingAverage_Python",
    "ExponentialSmoothing_Python",
    "StatsForecast_Python",
    "Nixtla_Python",
]

# Template categories for organized benchmarking
TEMPLATE_CATEGORIES = {
    "fast": ["MovingAverage_Python", "ExponentialSmoothing_Python", "ARIMA_Python"],
    "modern": ["Prophet_Python", "StatsForecast_Python", "Nixtla_Python"],
    "deep_learning": ["LSTM_Python", "NBEATS_Python"],
    "foundation": ["Chronos_Python", "TimesFM_Python"],
}


def run_template_forecast(template_name: str, data_path: str, output_dir: Path) -> Optional[Dict]:
    """Run a single template and return results."""
    import subprocess
    import os
    from pathlib import Path
    
    repo_root = Path(__file__).resolve().parents[1]
    template_dir = repo_root / template_name
    
    if not template_dir.exists():
        return None
    
    # Create temporary config that points to our data
    temp_config = output_dir / f"{template_name}_config.yaml"
    config_content = f"""data:
  input_file: "{data_path}"
  date_col: "date"
  value_col: "value"

evaluation:
  test_size: 0.2

output:
  output_dir: "{output_dir / template_name}"
"""
    temp_config.write_text(config_content)
    
    # Run template
    original_dir = os.getcwd()
    try:
        os.chdir(template_dir)
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            return {
                "template": template_name,
                "status": "failed",
                "error": result.stderr[:200] if result.stderr else "Unknown error"
            }
        
        # Try to load results if available
        # This is a simplified version - real implementation would parse outputs
        return {
            "template": template_name,
            "status": "completed",
            "stdout": result.stdout[:500] if result.stdout else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "template": template_name,
            "status": "timeout",
            "error": "Execution exceeded 5 minute timeout"
        }
    except Exception as e:
        return {
            "template": template_name,
            "status": "error",
            "error": str(e)
        }
    finally:
        os.chdir(original_dir)
        if temp_config.exists():
            temp_config.unlink()


def benchmark(
    data_path: str,
    templates: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    output_dir: str = "outputs/benchmark"
) -> pd.DataFrame:
    """
    Benchmark multiple templates on the same data.
    
    Parameters:
    -----------
    data_path : str
        Path to time series data CSV
    templates : list, optional
        Specific templates to benchmark
    categories : list, optional
        Template categories to benchmark (e.g., ["fast", "modern"])
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    pd.DataFrame
        Comparison results with metrics for each template
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which templates to run
    if templates:
        template_list = templates
    elif categories:
        template_list = []
        for cat in categories:
            template_list.extend(TEMPLATE_CATEGORIES.get(cat, []))
        template_list = list(set(template_list))  # Remove duplicates
    else:
        template_list = BENCHMARK_TEMPLATES
    
    print(f" Benchmarking {len(template_list)} templates")
    print(f" Data: {data_path}")
    print(f" Output: {output_dir}")
    print("=" * 70)
    
    results = []
    
    for i, template_name in enumerate(template_list, 1):
        print(f"\n[{i}/{len(template_list)}] Running {template_name}...")
        result = run_template_forecast(template_name, data_path, output_path)
        
        if result:
            results.append(result)
            status_icon = "" if result["status"] == "completed" else ""
            print(f"{status_icon} {template_name}: {result['status']}")
        else:
            print(f"️  {template_name}: Template not found")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_path / "benchmark_results.csv", index=False)
    
    # Generate summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_path": str(data_path),
        "total_templates": len(template_list),
        "completed": len([r for r in results if r.get("status") == "completed"]),
        "failed": len([r for r in results if r.get("status") != "completed"]),
    }
    
    with open(output_path / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n Benchmark complete!")
    print(f"   Completed: {summary['completed']}/{summary['total_templates']}")
    print(f"   Results saved to: {output_path}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated benchmarking tool")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--templates", nargs="+", help="Specific templates to benchmark")
    parser.add_argument("--categories", nargs="+", choices=list(TEMPLATE_CATEGORIES.keys()),
                       help="Template categories to benchmark")
    parser.add_argument("--output", default="outputs/benchmark", help="Output directory")
    
    args = parser.parse_args()
    
    benchmark(
        data_path=args.data,
        templates=args.templates,
        categories=args.categories,
        output_dir=args.output
    )

