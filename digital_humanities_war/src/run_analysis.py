"""Command line entry point for war sentiment workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import dataset_loaders, war_sentiment, plotting


def run_war_sentiment(config: dict, project_root: Path) -> None:
    dataset_cfg = dataset_loaders.DatasetConfig(
        path=project_root / config["dataset"]["path"],
        parser=config["dataset"].get("parser", "csv"),
        encoding=config["dataset"].get("encoding"),
    )
    analysis_cfg = war_sentiment.WarSentimentConfig(
        dataset=dataset_cfg,
        date_col=config.get("date_col", "date"),
        text_col=config.get("text_col", "text"),
        term_col=config.get("term_col", "term"),
        time_format=config.get("time_format"),
        aggregate=config.get("aggregate", "mean"),
    )

    result = war_sentiment.analyse(analysis_cfg)
    output_dir = project_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "war_sentiment_summary.csv"
    result.to_csv(csv_path, index=False)
    print(f"Saved aggregated sentiment to {csv_path}")

    figures_dir = output_dir / "figures"
    figure_paths = plotting.plot_term_trends(result, figures_dir, term_col=analysis_cfg.term_col)
    for path in figure_paths:
        print(f"Saved plot: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run war sentiment analysis")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        help="Override project root (defaults to package directory)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as fh:
        config = yaml.safe_load(fh)

    analysis_type = config.get("analysis")
    project_root = Path(args.project_root)

    if analysis_type == "war_sentiment":
        run_war_sentiment(config["war_sentiment"], project_root)
    else:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")


if __name__ == "__main__":
    main()
