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


def _slice_payload() -> dict[str, list[np.ndarray]]:
    return {
        "x": [],
        "y": [],
        "z": [],
        "C": [],
        "u": [],
        "v": [],
        "w": [],
        "p": [],
    }


def _append_masked_element(
    sliced: dict[str, list[np.ndarray]],
    element: Any,
    mask: np.ndarray,
) -> None:
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

    sliced["x"].append(x_flat[mask])
    sliced["y"].append(y_flat[mask])
    sliced["z"].append(z_flat[mask])
    sliced["C"].append(concentration_flat[mask])
    sliced["u"].append(u_flat[mask])
    sliced["v"].append(v_flat[mask])
    sliced["w"].append(w_flat[mask])
    sliced["p"].append(pressure_flat[mask])


def _collect_rounded_y_levels(data: Any, y_round_decimals: int) -> np.ndarray:
    rounded_y_values = []
    for element in data.elem:
        _, y, _ = get_coordinates(element)
        rounded_y_values.append(np.round(np.ravel(y), y_round_decimals))

    if not rounded_y_values:
        raise ValueError("No y coordinates found while selecting nearest y plane.")

    return np.unique(np.concatenate(rounded_y_values))


def extract_y_slice(
    data: Any,
    y0: float | None = None,
    slab_ratio: float = 0.01,
    mode: str = "nearest_plane",
    y_round_decimals: int = 10,
):
    """Extract a y-midspan slice using either one nearest y plane or a finite slab."""
    ymin, ymax = get_y_range(data)
    if y0 is None:
        y0 = 0.5 * (ymin + ymax)

    if mode not in {"nearest_plane", "slab"}:
        raise ValueError(f"Unknown slice mode {mode!r}. Expected 'nearest_plane' or 'slab'.")

    sliced = _slice_payload()
    metadata: dict[str, Any] = {
        "mode": mode,
        "y0": y0,
        "slab_ratio": slab_ratio,
        "y_round_decimals": y_round_decimals,
    }

    if mode == "nearest_plane":
        rounded_y_levels = _collect_rounded_y_levels(data, y_round_decimals)
        selected_y = float(rounded_y_levels[np.argmin(np.abs(rounded_y_levels - y0))])
        metadata.update(
            {
                "selected_y": selected_y,
                "rounded_unique_y_count_before_selection": int(rounded_y_levels.size),
            }
        )

        for element in data.elem:
            _, y, _ = get_coordinates(element)
            y_flat = np.ravel(y)
            mask = np.round(y_flat, y_round_decimals) == selected_y
            if not np.any(mask):
                continue
            _append_masked_element(sliced, element, mask)

        if not sliced["x"]:
            raise ValueError(f"No points found on nearest y plane selected_y={selected_y} around y0={y0}.")

    else:
        dy_tol = slab_ratio * (ymax - ymin)
        metadata["dy_tol"] = dy_tol

        for element in data.elem:
            _, y, _ = get_coordinates(element)
            y_flat = np.ravel(y)
            mask = np.abs(y_flat - y0) <= dy_tol
            if not np.any(mask):
                continue
            _append_masked_element(sliced, element, mask)

        if not sliced["x"]:
            raise ValueError(f"No points found in y slice around y0={y0} with dy_tol={dy_tol}. Increase slab_ratio.")

    slice_data: dict[str, Any] = {name: np.concatenate(values) for name, values in sliced.items()}
    metadata["point_count"] = slice_data["x"].size
    slice_data.update(metadata)
    return slice_data


def save_slice_npz(slice_data: dict[str, Any], output_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    """Save extracted slice arrays and metadata to a compressed NumPy archive."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {name: slice_data[name] for name in ("x", "y", "z", "C", "u", "v", "w", "p")}
    for name in (
        "mode",
        "y0",
        "selected_y",
        "dy_tol",
        "slab_ratio",
        "y_round_decimals",
        "rounded_unique_y_count_before_selection",
        "point_count",
    ):
        if name in slice_data:
            payload[name] = slice_data[name]

    if metadata:
        payload.update(metadata)

    np.savez_compressed(path, **payload)
