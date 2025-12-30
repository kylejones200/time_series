#!/usr/bin/env python3
"""
Greykite: Forecasting Library
LinkedIn's Greykite for flexible, powerful time series forecasting.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from greykite.framework.templates.autogen.forecast_config import (
    ForecastConfig,
    MetadataParam,
    ModelComponentsParam,
)
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Apply SignalPlot's clean defaults
signalplot.apply()


def repo_import(module: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj



ts_utils = repo_import("utils.ts_utils")
load_ts_data = ts_utils.load_ts_data


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(data, config):
    """Prepare data for Greykite."""
    df = pd.DataFrame({"ts": data.index, "y": data.values})

    regressor_cols = config["data"].get("regressors", [])
    [df.__setitem__(col, np.random.randn(len(df))) for col in regressor_cols]

    return df, regressor_cols


def create_forecaster(config):
    """Create Greykite forecaster."""
    forecaster = Forecaster()

    regressor_cols = config["data"].get("regressors", [])

    metadata_param = MetadataParam(
        time_col="ts",
        value_col="y",
        freq=None,
        regressor_cols=regressor_cols if regressor_cols else None,
    )

    model_components = (
        ModelComponentsParam(
            custom={
                "growth": {"growth_term": config["model"].get("growth_term", "linear")},
                "seasonality": {
                    "yearly_seasonality": config["model"].get(
                        "yearly_seasonality", "auto"
                    ),
                    "quarterly_seasonality": config["model"].get(
                        "quarterly_seasonality", "auto"
                    ),
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


def fit_and_predict(forecaster, forecast_config, df):
    """Fit model and generate predictions."""
    result = forecaster.run_forecast_config(df=df, config=forecast_config)
    return result.forecast.df


def create_visualizations(forecast_df, config):
    """Generate clean visualizations."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    historical = forecast_df[forecast_df["y"].notna()].tail(100)
    future = forecast_df[forecast_df["y"].isna()]

    fig, ax = plt.subplots(figsize=config)

    ax.plot(
        historical["ts"],
        historical["y"],
        c=config["plotting"]["style"]["colors"]["primary"],
        linewidth=config["plotting"]["style"]["linewidth"],
        alpha=config["plotting"]["style"]["alpha"],
        label="Historical",
    )

    ax.plot(
        historical["ts"],
        historical["forecast"],
        c=config["plotting"]["style"]["colors"]["accent"],
        linewidth=config["plotting"]["style"]["linewidth"] * 0.7,
        alpha=config["plotting"]["style"]["alpha"],
        label="Fitted",
    )

    ax.plot(
        future["ts"],
        future["forecast"],
        c=config["plotting"]["style"]["colors"]["secondary"],
        linewidth=config["plotting"]["style"]["linewidth"],
        label="Forecast",
    )

    [
        ax.fill_between(
            future["ts"],
            future["forecast_lower"],
            future["forecast_upper"],
            alpha=0.2,
            color=config["plotting"]["style"]["colors"]["secondary"],
        )
        for _ in [None]
        if "forecast_lower" in future.columns
    ]

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    plt.tight_layout()

    [
        fig.savefig(output_dir / "greykite_forecast.png", dpi=300, bbox_inches="tight", facecolor="white")
        for _ in [None]
        if config["output"]["save_plots"]
    ]
    plt.show()

    y_true = historical["y"]
    y_pred = historical["forecast"]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")


def main():
    """Main execution function."""
    config = load_config()
    data = load_ts_data(
        Path(__file__).parent.parent / "data" / config["data"]["input_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["value_col"],
    )

    df, regressor_cols = prepare_data(data, config)
    forecaster, forecast_config = create_forecaster(config)
    forecast_df = fit_and_predict(forecaster, forecast_config, df)
    create_visualizations(forecast_df, config)

    print("✓ Greykite forecasting complete")


if __name__ == "__main__":
    main()
