#!/usr/bin/env python3
"""
Sundial Transformer Forecasting
Forecast crude oil (or any FRED series) using the Sundial transformer model.
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from pandas_datareader.data import DataReader
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, save_plot, apply_plot_style, apply_legend

warnings.filterwarnings("ignore")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_fred_series(series_id: str, start_date: str, resample_rule: str | None) -> pd.Series:
    df = DataReader(series_id, "fred", start_date)
    df = df.rename(columns={series_id: "value"}).dropna()
    if resample_rule:
        df = df.resample(resample_rule).mean().dropna()
    return df["value"].astype(float)


def standardize(series: pd.Series) -> tuple[pd.Series, float, float]:
    mean = series.mean()
    std = series.std()
    if std == 0:
        raise ValueError("Standard deviation is zero; cannot normalize series.")
    normalized = (series - mean) / std
    return normalized, float(mean), float(std)


def load_sundial(model_name: str, token: str | None) -> AutoModelForCausalLM:
    kwargs = {"trust_remote_code": True}
    if token:
        kwargs["use_auth_token"] = token
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def generate_sundial_forecast(model: AutoModelForCausalLM, context: torch.Tensor, forecast_length: int,
                              num_samples: int, generation_kwargs: dict) -> torch.Tensor:
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            output = model.generate(
                context,
                max_new_tokens=forecast_length,
                do_sample=generation_kwargs.get("do_sample", True),
                temperature=generation_kwargs.get("temperature", 1.0),
                top_p=generation_kwargs.get("top_p", 0.9),
                top_k=generation_kwargs.get("top_k", 0),
            )
            samples.append(output.squeeze().float())
    return torch.stack(samples)


def invert_standardization(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return tensor * std + mean


def plot_results(config: dict, history: pd.Series, truth: np.ndarray | None,
                 mean_forecast: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> None:
    history_window = config['plotting'].get('history_window', len(history))
    history_segment = history.iloc[-history_window:]

    fig, ax = setup_figure(config['plotting']['figure_size'], config['plotting']['dpi'])
    apply_plot_style(ax, {'plotting': config['plotting']})

    ax.plot(history_segment.index, history_segment.values,
            color=config['plotting']['colors'][0], linewidth=config['plotting']['linewidth'],
            alpha=config['plotting']['alpha'], label='Historical')

    forecast_index = pd.date_range(start=history.index[-1], periods=len(mean_forecast) + 1, freq=history.index.freq)
    forecast_index = forecast_index[1:]

    ax.plot(forecast_index, mean_forecast.numpy(),
            color=config['plotting']['colors'][1], linestyle='--', linewidth=2,
            label='Sundial Forecast (mean)')

    ax.fill_between(forecast_index, lower.numpy(), upper.numpy(),
                    color=config['plotting']['colors'][1], alpha=0.2, label='80% interval')

    if truth is not None:
        ax.plot(forecast_index, truth, color=config['plotting']['colors'][2], linewidth=2, label='Actual')

    ax.set_title(config['plot_titles']['forecast'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    apply_legend(ax, config['plotting']['legend'])

    output_path = Path(__file__).parent / 'outputs' / 'sundial_forecast.png'
    save_plot(fig, output_path)
    plt.show()


def main():
    config = load_config()

    series = fetch_fred_series(
        config['data']['series_id'],
        config['data']['start_date'],
        config['data'].get('resample_rule')
    )

    series = series.asfreq(series.index.inferred_freq)
    normalized, mean, std = standardize(series)

    lookback = config['model']['lookback_length']
    forecast_length = config['model']['forecast_length']

    if len(normalized) < lookback + forecast_length:
        raise ValueError("Series is shorter than lookback + forecast length.")

    train_values = normalized.values[-(lookback + forecast_length):-forecast_length]
    true_future = series.values[-forecast_length:] if config['evaluation']['compare_to_actual'] else None

    context = torch.tensor(train_values, dtype=torch.float32).unsqueeze(0)

    hf_token = os.getenv('HF_TOKEN')
    model = load_sundial(config['model']['huggingface_model'], hf_token)

    samples = generate_sundial_forecast(
        model,
        context,
        forecast_length,
        config['model']['num_samples'],
        config['model'].get('generation', {})
    )

    samples = samples[:, -forecast_length:]
    mean_forecast = invert_standardization(samples.mean(dim=0), mean, std)
    lower = invert_standardization(samples.quantile(0.1, dim=0), mean, std)
    upper = invert_standardization(samples.quantile(0.9, dim=0), mean, std)

    forecast_df = pd.DataFrame({
        'date': pd.date_range(series.index[-forecast_length], periods=forecast_length + 1, freq=series.index.freq)[1:],
        'forecast_mean': mean_forecast.numpy(),
        'forecast_lower_80': lower.numpy(),
        'forecast_upper_80': upper.numpy(),
    })

    output_csv = Path(__file__).parent / 'outputs' / 'sundial_forecast.csv'
    forecast_df.to_csv(output_csv, index=False)
    print(f"Forecast saved to {output_csv}")

    if true_future is not None:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae = mean_absolute_error(true_future, mean_forecast.numpy())
        rmse = np.sqrt(mean_squared_error(true_future, mean_forecast.numpy()))
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    plot_results(config, series, true_future, mean_forecast, lower, upper)


if __name__ == "__main__":
    main()
