"""Input helpers for Nek5000 data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_nek_file(path: str | Path) -> Any:
    """Read a Nek5000 file.

    Parameters
    ----------
    path:
        Path to one Nek5000 field file.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Nek5000 file not found: {file_path}")

    from pymech.neksuite import readnek

    return readnek(str(file_path))


def get_nek_time(data: Any) -> Any:
    """Return the simulation time stored in a Nek5000 payload."""
    return getattr(data, "time", None)


def get_element_count(data: Any) -> int:
    """Return the number of elements contained in a Nek5000 payload."""
    return len(data.elem)


def get_first_element(data: Any) -> Any:
    """Return the first element in a Nek5000 payload."""
    if get_element_count(data) == 0:
        raise ValueError("Nek5000 payload contains no elements.")
    return data.elem[0]


def describe_nek_data(data: Any) -> dict[str, Any]:
    """Return basic metadata about a Nek5000 payload."""
    first_element = get_first_element(data)
    return {
        "time": get_nek_time(data),
        "element_count": get_element_count(data),
        "first_element_type": type(first_element).__name__,
        "first_element_attributes": [name for name in dir(first_element) if not name.startswith("_")],
    }
