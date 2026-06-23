"""Input helpers for Nek5000 data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_nek_file(path: str | Path) -> Any:
    """Read a Nek5000 file.

    This repository skeleton does not implement the actual reader yet.
    """
    raise NotImplementedError("Nek5000 file reading is not implemented yet.")


def get_nek_time(data: Any) -> float:
    """Return the simulation time stored in a Nek5000 payload."""
    raise NotImplementedError("Nek5000 time extraction is not implemented yet.")


def get_element_count(data: Any) -> int:
    """Return the number of elements contained in a Nek5000 payload."""
    raise NotImplementedError("Nek5000 element counting is not implemented yet.")
