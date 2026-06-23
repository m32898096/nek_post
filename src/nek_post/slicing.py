"""Spatial slicing helpers for Nek5000 data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nek_post.fields import get_concentration, get_coordinates, get_pressure, get_velocity


def get_y_range(data: Any):
    """Return the y-axis extent for a Nek5000 dataset."""
    ymin = np.inf
    ymax = -np.inf

    for element in data.elem:
        _, y, _ = get_coordinates(element)
        ymin = min(ymin, float(np.min(y)))
        ymax = max(ymax, float(np.max(y)))

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        raise ValueError("Could not determine y range because no coordinate points were found.")

    return ymin, ymax


def extract_y_slice(data: Any, y0: float | None = None, slab_ratio: float = 0.01):
    """Extract a midspan slice around `y0` with a given slab thickness."""
    ymin, ymax = get_y_range(data)
    if y0 is None:
        y0 = 0.5 * (ymin + ymax)

    dy_tol = slab_ratio * (ymax - ymin)

    sliced: dict[str, list[np.ndarray]] = {
        "x": [],
        "y": [],
        "z": [],
        "C": [],
        "u": [],
        "v": [],
        "w": [],
        "p": [],
    }

    for element in data.elem:
        x, y, z = get_coordinates(element)
        concentration = get_concentration(element)
        u, v, w = get_velocity(element)
        pressure = get_pressure(element)

        x_flat = np.ravel(x)
        y_flat = np.ravel(y)
        z_flat = np.ravel(z)
        concentration_flat = np.ravel(concentration)
        u_flat = np.ravel(u)
        v_flat = np.ravel(v)
        w_flat = np.ravel(w)
        pressure_flat = np.ravel(pressure)

        mask = np.abs(y_flat - y0) <= dy_tol
        if not np.any(mask):
            continue

        sliced["x"].append(x_flat[mask])
        sliced["y"].append(y_flat[mask])
        sliced["z"].append(z_flat[mask])
        sliced["C"].append(concentration_flat[mask])
        sliced["u"].append(u_flat[mask])
        sliced["v"].append(v_flat[mask])
        sliced["w"].append(w_flat[mask])
        sliced["p"].append(pressure_flat[mask])

    if not sliced["x"]:
        raise ValueError(f"No points found in y slice around y0={y0} with dy_tol={dy_tol}. Increase slab_ratio.")

    slice_data: dict[str, Any] = {name: np.concatenate(values) for name, values in sliced.items()}
    slice_data.update(
        {
            "y0": y0,
            "dy_tol": dy_tol,
            "slab_ratio": slab_ratio,
            "point_count": slice_data["x"].size,
        }
    )
    return slice_data


def save_slice_npz(slice_data: dict[str, Any], output_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    """Save extracted slice arrays and metadata to a compressed NumPy archive."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {name: slice_data[name] for name in ("x", "y", "z", "C", "u", "v", "w", "p")}
    for name in ("y0", "dy_tol", "slab_ratio", "point_count"):
        if name in slice_data:
            payload[name] = slice_data[name]

    if metadata:
        payload.update(metadata)

    np.savez_compressed(path, **payload)
