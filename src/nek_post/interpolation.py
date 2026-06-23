"""Interpolation helpers for projecting Nek5000 data onto common grids."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import griddata


def create_common_xz_grid(slice_data_by_case: dict[str, Any], nx: int, nz: int):
    """Create a common x-z grid for cross-order comparisons."""
    xmins = []
    xmaxs = []
    zmins = []
    zmaxs = []

    for case, slice_data in slice_data_by_case.items():
        x = np.asarray(slice_data["x"], dtype=float)
        z = np.asarray(slice_data["z"], dtype=float)
        if x.size == 0 or z.size == 0:
            raise ValueError(f"Slice data for {case} contains no x-z points.")

        xmins.append(float(np.nanmin(x)))
        xmaxs.append(float(np.nanmax(x)))
        zmins.append(float(np.nanmin(z)))
        zmaxs.append(float(np.nanmax(z)))

    xmin = max(xmins)
    xmax = min(xmaxs)
    zmin = max(zmins)
    zmax = min(zmaxs)

    if xmin >= xmax or zmin >= zmax:
        raise ValueError("Slice files do not share an overlapping x-z domain.")

    xi = np.linspace(xmin, xmax, nx)
    zi = np.linspace(zmin, zmax, nz)
    Xi, Zi = np.meshgrid(xi, zi)
    metadata = {
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "nx": nx,
        "nz": nz,
    }
    return Xi, Zi, xi, zi, metadata


def interpolate_to_grid(x, z, values, Xi, Zi, method: str = "linear"):
    """Interpolate values defined on scattered x-z points to a structured grid."""
    points = (np.asarray(x, dtype=float), np.asarray(z, dtype=float))
    return griddata(points, np.asarray(values, dtype=float), (Xi, Zi), method=method)


def valid_common_mask(*arrays):
    """Return a mask where all input arrays are finite."""
    if not arrays:
        raise ValueError("At least one array is required to build a valid mask.")

    mask = np.ones_like(np.asarray(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask
