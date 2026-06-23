"""Configuration loading helpers for the Nek5000 post-processing project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its parsed contents."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_project_config(paths_file: str | Path, cases_file: str | Path) -> dict[str, Any]:
    """Load the path and case configuration files into a single mapping."""
    return {
        "paths": load_yaml(paths_file),
        "cases": load_yaml(cases_file),
    }
