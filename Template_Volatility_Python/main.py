#!/usr/bin/env python3
"""
Volatility Models (ARCH/GARCH)
Volatility forecasting using ARCH and GARCH models for financial time series.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signalplot
from pathlib import Path
from importlib import util
from arch import arch_model
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
ensure_datetime_index = ts_utils.ensure_datetime_index
split_ts = ts_utils.split_ts


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_volatility_model(data, config):
    """Create ARCH/GARCH volatility model."""
    model_type = config["model"]["type"]

    model_map = {
        "ARCH": lambda: arch_model(
            data,
            vol=model_type,
            p=config["model"]["p"],
            q=0,
            dist=config["model"]["distribution"],
        ),
        "GARCH": lambda: arch_model(
            data,
            vol=model_type,
            p=config["model"]["p"],
            q=config["model"]["q"],
            dist=config["model"]["distribution"],
        ),
        "EGARCH": lambda: arch_model(
            data,
            vol="EGARCH",
            p=config["model"]["p"],
            q=config["model"]["q"],
            dist=config["model"]["distribution"],
        ),
    }

    return model_map.get(model_type, model_map["GARCH"])()


def fit_and_forecast(model, config):
    """Fit model and generate volatility forecasts."""
    fitted_model = model.fit(
        update_freq=config["model"].get("update_freq", 1),
        disp=config["model"].get("disp", "off"),
    )

    forecast = fitted_model.forecast(horizon=config["model"]["forecast_horizon"])
    forecast_variance = forecast.variance.iloc[-1].values
    forecast_volatility = np.sqrt(forecast_variance)

    return fitted_model, forecast_variance, forecast_volatility


def create_visualizations(
    data,
    train_data,
    test_data,
    fitted_model,
    forecast_variance,
    forecast_volatility,
    config,
):
    """Generate visualizations for volatility model."""
    output_dir = Path(__file__).parent / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for ax in axes.flatten():
        
    axes[0, 0].plot(
        train_data.index,
        train_data.values,
        "k-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label="Train Returns",
    )
    if test_data is not None:
        axes[0, 0].plot(
            test_data.index,
            test_data.values,
            "g-",
            linewidth=config["plotting"]["linewidth"],
            alpha=config["plotting"]["alpha"],
            label="Test Returns",
        )
    axes[0, 0].set_title("Time Series Returns")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Returns")
        axes[0, 0].legend()

    conditional_vol = fitted_model.conditional_volatility
    axes[0, 1].plot(
        train_data.index,
        conditional_vol,
        "r-",
        linewidth=config["plotting"]["linewidth"],
        alpha=config["plotting"]["alpha"],
        label="Conditional Volatility",
    )
    axes[0, 1].set_title("Conditional Volatility (Fitted)")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Volatility")
        axes[0, 1].legend()

    forecast_index = pd.date_range(
        start=train_data.index[-1] + pd.Timedelta(days=1),
        periods=len(forecast_variance),
        freq="D",
    )

    axes[1, 0].plot(
        forecast_index,
        forecast_variance,
        "b-",
        linewidth=config["plotting"]["linewidth"],
        marker="o",
        markersize=config["plotting"]["markersize"],
        label="Forecasted Variance",
    )
    axes[1, 0].set_title("Forecasted Variance")
    axes[1, 0].set_xlabel("Forecast Horizon")
    axes[1, 0].set_ylabel("Variance")
        axes[1, 0].legend()

    axes[1, 1].plot(
        forecast_index,
        forecast_volatility,
        "m-",
        linewidth=config["plotting"]["linewidth"],
        marker="o",
        markersize=config["plotting"]["markersize"],
        label="Forecasted Volatility",
    )
    axes[1, 1].set_title("Forecasted Volatility (sqrt of variance)")
    axes[1, 1].set_xlabel("Forecast Horizon")
    axes[1, 1].set_ylabel("Volatility")
        axes[1, 1].legend()

    plt.tight_layout()

    output_path = output_dir / "volatility_forecast.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def main():
    """Main execution function."""
    config = load_config()

    df = load_ts_data(
        data_path=Path(__file__).parent.parent / "data" / config["data"]["input_file"],
        date_col=config["data"]["date_col"],
        value_col=config["data"]["value_col"],
    )
    df = ensure_datetime_index(df, time_col=config["data"]["date_col"])

    if config["data"]["compute_returns"]:
        returns = df[config["data"]["value_col"]].pct_change().dropna()
    else:
        returns = df[config["data"]["value_col"]]

    train_returns, test_returns = split_ts(
        returns, test_size=config["data"]["test_size"]
    )

    model = create_volatility_model(train_returns.values, config)

    print(f"\nFitting {config['model']['type']} model...")
    fitted_model, forecast_variance, forecast_volatility = fit_and_forecast(
        model, config
    )

    print("\nVolatility Model Results:")
    print("=" * 70)
    print(fitted_model.summary())

    print(f"\nForecast Statistics:")
    print(f"Mean Forecasted Variance: {np.mean(forecast_variance):.6f}")
    print(f"Mean Forecasted Volatility: {np.mean(forecast_volatility):.6f}")
    print(f"Forecast Horizon: {config['model']['forecast_horizon']} steps")

    create_visualizations(
        returns,
        train_returns,
        test_returns,
        fitted_model,
        forecast_variance,
        forecast_volatility,
        config,
    )

    print(f"✓ {config['model']['type']} volatility forecasting complete")


if __name__ == "__main__":
    main()
