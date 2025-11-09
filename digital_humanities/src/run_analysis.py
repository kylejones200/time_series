"""Command line entry point for running configurable analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from . import dataset_loaders, sentiment_analysis, plotting


def run_sentiment(config: dict, base_dir: Path) -> None:
    dataset_cfg = dataset_loaders.DatasetConfig(
        path=base_dir / config["dataset"]["path"],
        parser=config["dataset"].get("parser", "csv"),
        encoding=config["dataset"].get("encoding"),
    )
    sentiment_cfg = sentiment_analysis.SentimentConfig(
        dataset=dataset_cfg,
        date_col=config["date_col"],
        value_col=config["value_col"],
        smoothing_window=config.get("smoothing_window", 5),
    )
    df = sentiment_analysis.load_sentiment(sentiment_cfg)
    output_dir = base_dir / config["output_dir"]
    summary_path = sentiment_analysis.export_summary(df, output_dir)
    print(f"Saved summary to {summary_path}")
    plot_path = output_dir / "trend.png"
    plotting.plot_trend(
        df,
        x=sentiment_cfg.date_col,
        y=sentiment_cfg.value_col,
        smoothed_col=f"{sentiment_cfg.value_col}_smoothed",
        output_path=plot_path,
        title=config.get("title", "Sentiment Trend"),
    )
    print(f"Saved plot to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run digital humanities analysis")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    analysis_type = config.get("analysis")
    project_root = Path(args.project_root)

    if analysis_type == "sentiment":
        run_sentiment(config["sentiment"], project_root)
    else:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")


if __name__ == "__main__":
    main()
