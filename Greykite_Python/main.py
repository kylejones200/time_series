#!/usr/bin/env python3
"""
Greykite: Forecasting Library
LinkedIn's Greykite for flexible, powerful time series forecasting.
"""

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)
from src.evaluator import Evaluator
from src.run_logger import append_run_log, utc_now_iso

from greykite.framework.templates.autogen.forecast_config import (
    ForecastConfig,
    MetadataParam,
    ModelComponentsParam,
)
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_data(data: pd.Series, config: dict):
    """Prepare data for Greykite."""
    df = pd.DataFrame({"ts": data.index, "y": data.values})
    
    regressor_cols = config["data"].get("regressors", [])
    for col in regressor_cols:
        df[col] = np.random.randn(len(df))
    
    return df, regressor_cols


def create_forecaster(config: dict):
    """Create Greykite forecaster."""
    forecaster = Forecaster()
    
    regressor_cols = config["data"].get("regressors", [])
    
    metadata_param = MetadataParam(
        time_col="ts",
        value_col="y",
        freq=None,
    )
    
    model_components = (
        ModelComponentsParam(
            custom={
                "growth": {"growth_term": config["model"].get("growth_term", "linear")},
                "seasonality": {
                    "yearly_seasonality": config["model"].get("yearly_seasonality", "auto"),
                    "quarterly_seasonality": config["model"].get("quarterly_seasonality", "auto"),
                },
                "extra_pred_cols": regressor_cols if regressor_cols else [],
            }
        )
        if regressor_cols or config["model"].get("seasonality")
        else None
    )
    
    forecast_config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=config["model"]["forecast_horizon"],
        coverage=config["model"].get("coverage", 0.95),
        metadata_param=metadata_param,
        model_components_param=model_components,
    )
    
    return forecaster, forecast_config


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    started_at = utc_now_iso()
    t0 = time.perf_counter()
    config = load_config()
    output_dir = ensure_output_dir(get_output_dir(config, script_dir))
    status = "success"
    error_msg = None
    metrics_log: dict[str, float] = {}

    try:
        # Load data using consolidated loader
        series = load_time_series(
            config["data"]["input_file"],
            date_column=config["data"].get("date_col", "date"),
            value_column=config["data"].get("value_col", "value")
        )

        print(f"Loaded {len(series)} data points")

        # Split train/test using consolidated evaluator
        evaluator = Evaluator(test_size=config.get("evaluation", {}).get("test_size", 0.2))
        train, test = evaluator.split(series)
        print(f"\nTrain: {len(train)} points, Test: {len(test)} points")

        # Prepare data for Greykite
        df, regressor_cols = prepare_data(train, config)

        # Create and fit forecaster
        print("\nCreating Greykite forecaster...")
        forecaster, forecast_config = create_forecaster(config)

        print("Fitting Greykite model...")
        result = forecaster.run_forecast_config(df=df, config=forecast_config)

        # Generate forecast
        forecast = result.forecast.df
        metrics_log["forecast_periods"] = float(len(forecast))
        print(f"\nGenerated forecast for {len(forecast)} periods")

        # Evaluate
        if len(test) > 0:
            forecast_values = forecast["forecast"].values[:len(test)]
            test_values = test.values[:len(forecast_values)]

            mae_val = mean_absolute_error(test_values, forecast_values)
            rmse_val = np.sqrt(mean_squared_error(test_values, forecast_values))
            metrics_log["mae"] = float(mae_val)
            metrics_log["rmse"] = float(rmse_val)

            print(f"\nEvaluation Metrics:")
            print(f"  MAE: {mae_val:.4f}")
            print(f"  RMSE: {rmse_val:.4f}")

        # Create visualization
        print("\nCreating visualization...")
        fig, ax = plt.subplots(figsize=config.get("plotting", {}).get("figure_size", [12, 6]))

        ax.plot(
            train.index[-100:] if len(train) > 100 else train.index,
            train.values[-100:] if len(train) > 100 else train.values,
            "k-",
            linewidth=config.get("plotting", {}).get("linewidth", 1.5),
            alpha=config.get("plotting", {}).get("alpha", 0.8),
            label="Historical",
        )

        if len(test) > 0:
            ax.plot(
                test.index[:len(forecast)] if len(test) >= len(forecast) else test.index,
                test.values[:len(forecast)] if len(test) >= len(forecast) else test.values,
                "g-",
                linewidth=config.get("plotting", {}).get("linewidth", 1.5),
                alpha=config.get("plotting", {}).get("alpha", 0.8),
                label="Actual (Test)",
            )

        ax.plot(
            forecast["ts"],
            forecast["forecast"],
            "r--",
            linewidth=config.get("plotting", {}).get("linewidth", 1.5),
            label="Greykite Forecast",
        )

        if "forecast_lower" in forecast.columns and "forecast_upper" in forecast.columns:
            ax.fill_between(
                forecast["ts"],
                forecast["forecast_lower"],
                forecast["forecast_upper"],
                alpha=0.2,
                color="r",
                label="95% CI",
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title("Greykite Forecast")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if config.get("output", {}).get("save_plots", True):
            save_plot(fig, output_dir / "greykite_forecast.png", dpi=300)
            print(f"Plot saved to: {output_dir / 'greykite_forecast.png'}")

        print("\n Greykite forecasting complete")

        if config.get("plotting", {}).get("show_plot", True):
            plt.show()
        else:
            plt.close(fig)
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        raise
    finally:
        ended_at = utc_now_iso()
        duration = time.perf_counter() - t0
        log_path = append_run_log(
            output_dir=output_dir,
            script_name="Greykite_Python",
            started_at_utc=started_at,
            ended_at_utc=ended_at,
            duration_seconds=duration,
            status=status,
            metrics=metrics_log,
            details={"data_path": str(config["data"]["input_file"])},
            error=error_msg,
        )
        print(f"Run log saved to: {log_path}")


if __name__ == "__main__":
    main()
