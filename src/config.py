#!/usr/bin/env python3
"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


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
    
    # Normalize data paths relative to repo root
    config_dir = config_file.parent
    repo_root = config_dir.parent  # Assuming config.yaml is one level from repo root
    
    if "data" in config and "input_file" in config["data"]:
        input_file = config["data"]["input_file"]
        # If not absolute, resolve relative to repo root
        if not Path(input_file).is_absolute():
            config["data"]["input_file"] = str(repo_root / "data" / input_file)
    
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

