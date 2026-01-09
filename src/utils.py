#!/usr/bin/env python3
"""Shared utility functions for templates."""

from pathlib import Path
from importlib import util
from typing import Any


def repo_import(module: str) -> Any:
    """
    Import module from repository root.
    
    This is the consolidated version of repo_import that was duplicated
    in 37+ templates. Use this instead of defining it in each template.
    
    Parameters:
    -----------
    module : str
        Module path relative to repo root (e.g., "utils.ts_utils")
        
    Returns:
    --------
    Module object
    """
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    spec = util.spec_from_file_location(module, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module '{module}' from {module_path}")
    module_obj = util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj


def ensure_output_dir(output_dir: Path, exist_ok: bool = True) -> Path:
    """
    Ensure output directory exists.
    
    Parameters:
    -----------
    output_dir : Path
        Directory path to create
    exist_ok : bool
        If True, don't raise error if directory exists
        
    Returns:
    --------
    Path
        Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=exist_ok)
    return output_dir

