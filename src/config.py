#!/usr/bin/env python3
"""Configuration loading utilities."""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ensure_output_dir


@dataclass
class CommonConfig:
    """Fields shared across template-specific config dataclasses."""

    data_path: Path
    date_col: str
    value_col: str
    output_dir: Path
    repo_root: Path
    script_dir: Path


def parse_common_config(
    config_dict: Dict[str, Any],
    script_dir: Path,
    *,
    output_dir: Optional[Path] = None,
) -> CommonConfig:
    """
    Parse shared config fields used by most forecasting templates.

    Consolidates the repeated data_path / date_col / value_col / output_dir
    logic that was copy-pasted across template parse_config() functions.
    """
    script_dir = Path(script_dir)
    repo_root = script_dir.parent
    data_cfg = config_dict.get("data", {})

    input_file = data_cfg.get("input_file", "")
    input_path = Path(input_file)
    if input_path.is_absolute():
        data_path = input_path
    else:
        candidate = repo_root / "data" / input_file
        data_path = candidate if candidate.exists() else input_path

    if output_dir is None:
        output_name = config_dict.get("output", {}).get("output_dir", "outputs")
        output_dir = ensure_output_dir(script_dir / output_name)

    date_col = data_cfg.get("date_col") or data_cfg.get("date_column", "date")
    value_col = data_cfg.get("value_col") or data_cfg.get("value_column", "value")

    return CommonConfig(
        data_path=data_path,
        date_col=date_col,
        value_col=value_col,
        output_dir=output_dir,
        repo_root=repo_root,
        script_dir=script_dir,
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Unified config loader that handles path resolution relative to config file.
    This consolidates the 56+ variations of load_config across templates.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Use explicit encoding and ensure file is properly closed on Windows
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve data file paths to absolute using this repo's data/ directory.
    # config.py lives at <repo_root>/src/config.py, so parent.parent = <repo_root>.
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    if "data" in config:
        for key in ("input_file", "series1_file", "series2_file"):
            if key in config["data"]:
                fname = config["data"][key]
                if fname and not Path(fname).is_absolute():
                    candidate = data_dir / fname
                    if candidate.exists():
                        config["data"][key] = str(candidate)

    return config


def get_output_dir(config: Dict[str, Any], script_dir: Path) -> Path:
    """
    Get output directory path from config.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    script_dir : Path
        Directory where script is located (usually __file__.parent)
        
    Returns:
    --------
    Path
        Output directory path
    """
    output_dir_name = config.get("output", {}).get("output_dir", "outputs")
    return script_dir / output_dir_name

