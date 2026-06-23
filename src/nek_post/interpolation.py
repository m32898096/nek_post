"""Interpolation helpers for projecting Nek5000 data onto common grids."""

from __future__ import annotations


def create_common_xz_grid(x, z, nx: int, nz: int):
    """Create a common x-z grid for cross-order comparisons."""
    raise NotImplementedError


def interpolate_to_grid(x, z, values, xi, zi):
    """Interpolate values defined on scattered x-z points to a structured grid."""
    raise NotImplementedError
