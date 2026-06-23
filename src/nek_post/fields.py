"""Field access helpers for Nek5000 data."""

from __future__ import annotations

from typing import Any


def get_coordinates(element: Any):
    """Return coordinate arrays for one element."""
    raise NotImplementedError


def get_concentration(element: Any):
    """Return the concentration field for one element."""
    raise NotImplementedError


def get_velocity(element: Any):
    """Return the velocity field for one element."""
    raise NotImplementedError


def get_pressure(element: Any):
    """Return the pressure field for one element."""
    raise NotImplementedError


def get_speed(u, v, w):
    """Compute speed magnitude from velocity components."""
    raise NotImplementedError


def subtract_mean_pressure(p):
    """Return pressure with its mean removed."""
    raise NotImplementedError
