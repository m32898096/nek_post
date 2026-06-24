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


def average_duplicate_xz_points(x, z, values, decimals: int = 10):
    """Average values that share the same rounded projected x-z location."""
    x_arr = np.asarray(x, dtype=float).ravel()
    z_arr = np.asarray(z, dtype=float).ravel()
    values_arr = np.asarray(values, dtype=float).ravel()
    if x_arr.shape != z_arr.shape or x_arr.shape != values_arr.shape:
        raise ValueError("x, z, and values must have matching one-dimensional sizes.")

    finite_mask = np.isfinite(x_arr) & np.isfinite(z_arr) & np.isfinite(values_arr)
    if not np.any(finite_mask):
        raise ValueError("No finite x-z-value points are available for duplicate averaging.")

    x_finite = x_arr[finite_mask]
    z_finite = z_arr[finite_mask]
    values_finite = values_arr[finite_mask]
    keys = np.column_stack((np.round(x_finite, decimals), np.round(z_finite, decimals)))
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)

    x_sum = np.bincount(inverse, weights=x_finite)
    z_sum = np.bincount(inverse, weights=z_finite)
    values_sum = np.bincount(inverse, weights=values_finite)

    stats = {
        "original_point_count": int(x_arr.size),
        "finite_point_count": int(x_finite.size),
        "unique_point_count": int(counts.size),
        "duplicate_point_count": int(x_finite.size - counts.size),
        "max_multiplicity": int(np.max(counts)),
        "decimals": int(decimals),
    }
    return x_sum / counts, z_sum / counts, values_sum / counts, stats


def interpolate_to_grid(
    x,
    z,
    values,
    Xi,
    Zi,
    method: str = "linear",
    deduplicate: bool = True,
    duplicate_decimals: int = 10,
):
    """Interpolate values defined on scattered x-z points to a structured grid."""
    if deduplicate:
        x_interp, z_interp, values_interp, _ = average_duplicate_xz_points(
            x,
            z,
            values,
            decimals=duplicate_decimals,
        )
    else:
        x_interp = np.asarray(x, dtype=float).ravel()
        z_interp = np.asarray(z, dtype=float).ravel()
        values_interp = np.asarray(values, dtype=float).ravel()

    points = (x_interp, z_interp)
    return griddata(points, values_interp, (Xi, Zi), method=method)


def valid_common_mask(*arrays):
    """Return a mask where all input arrays are finite."""
    if not arrays:
        raise ValueError("At least one array is required to build a valid mask.")

    mask = np.ones_like(np.asarray(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask
