#!/usr/bin/env python3
"""TimesFM forecasting template aligned with the 2025-11-08 article assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timesfm
import yaml
from sklearn.metrics import mean_absolute_error


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    freq: str
    history_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    checkpoint: str
    backend: str
    per_core_batch_size: int
    context_length: int
    horizon_length: int
    output_dir: Path
    output_plot: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg["data"]["input_file"]
    output_dir = Path(__file__).parent / cfg["output"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    experiment = cfg["experiment"]
    model_cfg = cfg["model"]

    return Config(
        data_path=data_path,
        date_col=cfg["data"]["date_col"],
        value_col=cfg["data"]["value_col"],
        freq=cfg["data"].get("frequency", "M"),
        history_end=pd.Timestamp(experiment["history_end"]),
        forecast_start=pd.Timestamp(experiment["forecast_start"]),
        forecast_end=pd.Timestamp(experiment["forecast_end"]),
        checkpoint=model_cfg["checkpoint"],
        backend=model_cfg.get("backend", "cpu"),
        per_core_batch_size=int(model_cfg.get("per_core_batch_size", 32)),
        context_length=int(model_cfg.get("context_length", 512)),
        horizon_length=int(model_cfg.get("horizon_length", 8)),
        output_dir=output_dir,
        output_plot=output_dir / cfg["output"]["tufte_plot"],
    )


def load_series(config: Config) -> pd.Series:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {config.data_path}")

    df = pd.read_csv(config.data_path)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not present in CSV")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.value_col])
    df = df.sort_values(config.date_col).set_index(config.date_col)
    series = pd.to_numeric(df[config.value_col], errors="coerce").dropna()
    return series.asfreq(config.freq).astype(float)


def build_model(config: Config) -> timesfm.TimesFm:
    hparams_kwargs = {
        "backend": config.backend,
        "per_core_batch_size": config.per_core_batch_size,
        "horizon_len": config.horizon_length,
    }
    if "context_length" in timesfm.TimesFmHparams.__dataclass_fields__:
        hparams_kwargs["context_length"] = config.context_length
    hparams = timesfm.TimesFmHparams(**hparams_kwargs)
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=config.checkpoint)
    return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)


def prepare_training_frame(series: pd.Series, end: pd.Timestamp) -> pd.DataFrame:
    train = series.loc[:end]
    return pd.DataFrame({"unique_id": ["EIA"] * len(train), "ds": train.index, "y": train.values})


def run_timesfm_forecast(model: timesfm.TimesFm, df: pd.DataFrame, freq: str) -> pd.DataFrame:
    fc = model.forecast_on_df(inputs=df, freq=freq, value_name="y", num_jobs=-1)
    fc = fc[fc["unique_id"] == df["unique_id"].iloc[0]].copy()
    fc = fc.set_index("ds").sort_index()
    return fc


def select_forecast_column(df: pd.DataFrame) -> str:
    for candidate in ["y_hat", "yhat", "mean", "y", "point_forecast"]:
        if candidate in df.columns:
            return candidate
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]
    raise RuntimeError(f"No numeric forecast column found in TimesFM output: {list(df.columns)}")


def compute_metrics(actual: pd.Series, forecast: pd.Series) -> Tuple[float, float]:
    aligned = actual.align(forecast, join="inner")
    if aligned[0].empty:
        return float("nan"), float("nan")
    mae = mean_absolute_error(aligned[0], aligned[1])
    denom = np.where(aligned[0] == 0, np.nan, aligned[0])
    mape = np.nanmean(np.abs((aligned[0] - aligned[1]) / denom)) * 100
    return mae, float(mape)


def plot_tufte(series: pd.Series, history_end: pd.Timestamp, forecast_series: pd.Series, actual: pd.Series, config: Config) -> None:
    start_2024 = pd.Timestamp("2024-01-01")
    history = series.loc[start_2024:history_end]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, color="#888888", lw=1.5)
    ax.axvline(config.forecast_start, color="#666666", linestyle="--", lw=1)
    if not actual.empty:
        ax.plot(actual.index, actual.values, color="#444444", lw=1.8)
    if not forecast_series.empty:
        ax.plot(forecast_series.index, forecast_series.values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_title("EIA Net Generation — TimesFM forecast Jan–Aug 2025")

    if not history.empty:
        ax.annotate(
            "History (2024)",
            xy=(history.index[-1], history.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#666666",
        )
    if not actual.empty:
        ax.annotate(
            "Actual (Jan–Aug 2025)",
            xy=(actual.index[-1], actual.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#444444",
        )
    if not forecast_series.empty:
        ax.annotate(
            "TimesFM",
            xy=(forecast_series.index[-1], forecast_series.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#000000",
        )

    fig.tight_layout()
    fig.savefig(config.output_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ TimesFM plot saved -> {config.output_plot}")


def main() -> None:
    config = load_config()
    series = load_series(config)

    actual = series.loc[config.forecast_start : config.forecast_end]
    train_df = prepare_training_frame(series, config.history_end)

    model = build_model(config)
    forecast_df = run_timesfm_forecast(model, train_df, config.freq)
    forecast_col = select_forecast_column(forecast_df)
    forecast_slice = forecast_df.loc[config.forecast_start : config.forecast_end, forecast_col]

    mae, mape = compute_metrics(actual, forecast_slice)
    if not np.isnan(mae):
        print(f"TimesFM MAE (Jan–Aug 2025): {mae:,.2f}")
        print(f"TimesFM MAPE (Jan–Aug 2025): {mape:,.2f}%")

    plot_tufte(series, config.history_end, forecast_slice, actual, config)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""TimesFM forecasting template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timesfm
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plotting_utils import setup_figure, apply_plot_style, apply_legend, save_plot


@dataclass
class Config:
    data_path: Path
    date_col: str
    value_col: str
    frequency: str
    prediction_length: int
    forecast_column: str
    model_checkpoint: str
    per_core_batch_size: int
    input_patch_len: int
    output_patch_len: int
    horizon_len: int
    num_layers: int
    model_dims: int
    use_positional_embedding: bool
    output_dir: Path


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / cfg['data']['input_file']
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    model_cfg = cfg['model']
    return Config(
        data_path=data_path,
        date_col=cfg['data']['date_col'],
        value_col=cfg['data']['value_col'],
        frequency=cfg['data']['frequency'],
        prediction_length=cfg['model']['prediction_length'],
        forecast_column=model_cfg.get('forecast_column', 'timesfm'),
        model_checkpoint=model_cfg['checkpoint'],
        per_core_batch_size=model_cfg.get('per_core_batch_size', 32),
        input_patch_len=model_cfg.get('input_patch_len', 32),
        output_patch_len=model_cfg.get('output_patch_len', 128),
        horizon_len=model_cfg.get('horizon_len', 128),
        num_layers=model_cfg.get('num_layers', 50),
        model_dims=model_cfg.get('model_dims', 1280),
        use_positional_embedding=model_cfg.get('use_positional_embedding', False),
        output_dir=output_dir,
    )


def load_series(config: Config) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found at {config.data_path}. "
            "Provide a dataset with date and value columns."
        )

    df = pd.read_csv(config.data_path)
    if config.date_col not in df.columns or config.value_col not in df.columns:
        raise ValueError("Specified columns not present in CSV")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors='coerce')
    df = df.dropna(subset=[config.date_col, config.value_col])

    df_grouped = df.groupby(config.date_col)[config.value_col].sum().reset_index()
    df_grouped = df_grouped.sort_values(config.date_col)
    df_grouped['unique_id'] = 'series_1'
    df_grouped.rename(columns={config.date_col: 'ds', config.value_col: 'y'}, inplace=True)
    return df_grouped[['unique_id', 'ds', 'y']]


def split_train_test(df: pd.DataFrame, prediction_length: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= prediction_length:
        raise ValueError("Prediction length must be smaller than the series length")
    train = df.iloc[:-prediction_length].copy()
    test = df.iloc[-prediction_length:].copy()
    return train, test


def build_model(config: Config) -> timesfm.TimesFm:
    hparams = timesfm.TimesFmHparams(
        per_core_batch_size=config.per_core_batch_size,
        horizon_len=config.horizon_len,
        input_patch_len=config.input_patch_len,
        output_patch_len=config.output_patch_len,
        num_layers=config.num_layers,
        model_dims=config.model_dims,
        use_positional_embedding=config.use_positional_embedding,
    )
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=config.model_checkpoint)
    model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    return model


def run_forecast(model: timesfm.TimesFm, train_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    forecast_df = model.forecast_on_df(
        inputs=train_df,
        freq=config.frequency,
        num_jobs=-1,
    )
    if config.forecast_column not in forecast_df.columns:
        raise KeyError(
            f"Forecast column '{config.forecast_column}' not found in TimesFM output. "
            f"Available columns: {forecast_df.columns.tolist()}"
        )
    return forecast_df[['ds', config.forecast_column]].copy()


def compute_metrics(test_df: pd.DataFrame, forecast_df: pd.DataFrame, prediction_length: int,
                    config: Config) -> tuple[np.ndarray, dict]:
    preds = forecast_df[config.forecast_column].values[-prediction_length:]
    truth = test_df['y'].values

    mae = mean_absolute_error(truth, preds)
    rmse = np.sqrt(mean_squared_error(truth, preds))
    mape = np.mean(np.abs((truth - preds) / truth)) * 100
    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    return preds, metrics


def plot_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, preds: np.ndarray,
                  metrics: dict, config: Config) -> None:
    full_series = pd.concat([train_df, test_df], ignore_index=True)
    fig, ax = setup_figure((14, 6), 150)
    apply_plot_style(ax, {'plotting': {
        'style': {
            'spines': {'top': False, 'right': False, 'bottom': True, 'left': True},
            'grid': False
        }
    }})

    ax.plot(full_series['ds'], full_series['y'], color='black', label='Actual', linewidth=2)
    ax.plot(test_df['ds'], preds, color='tomato', label='TimesFM Forecast', linewidth=2)

    ax.set_title('TimesFM Forecast', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

    metrics_text = '\n'.join(f"{k}: {v:.3f}" for k, v in metrics.items())
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    apply_legend(ax, {'frameon': False, 'loc': 'best'})
    output_path = config.output_dir / 'timesfm_forecast.png'
    save_plot(fig, output_path)
    plt.close(fig)
    print(f"✓ Forecast plot saved -> {output_path}")


def save_outputs(test_df: pd.DataFrame, preds: np.ndarray, metrics: dict, config: Config) -> None:
    forecast_df = pd.DataFrame({
        'ds': test_df['ds'].values,
        'actual': test_df['y'].values,
        'forecast': preds,
    })
    forecast_path = config.output_dir / 'timesfm_forecast.csv'
    forecast_df.to_csv(forecast_path, index=False)

    metrics_path = config.output_dir / 'timesfm_metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.safe_dump(metrics, f)

    print(f"✓ Forecast data saved -> {forecast_path}")
    print(f"✓ Metrics saved -> {metrics_path}")


def main():
    config = load_config()
    series_df = load_series(config)
    train_df, test_df = split_train_test(series_df, config.prediction_length)

    model = build_model(config)
    forecast_df = run_forecast(model, train_df, config)

    preds, metrics = compute_metrics(test_df, forecast_df, config.prediction_length, config)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    plot_forecast(train_df, test_df, preds, metrics, config)
    save_outputs(test_df, preds, metrics, config)


if __name__ == "__main__":
    main()
