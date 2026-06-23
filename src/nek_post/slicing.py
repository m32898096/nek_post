"""Spatial slicing helpers for Nek5000 data."""

from __future__ import annotations

from typing import Any


def get_y_range(data: Any):
    """Return the y-axis extent for a Nek5000 dataset."""
    raise NotImplementedError


def extract_y_slice(data: Any, y0: float, slab_ratio: float):
    """Extract a midspan slice around `y0` with a given slab thickness."""
    raise NotImplementedError
