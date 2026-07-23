"""Fixed-grid concentration sequence construction for front detection."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from nek_post.front_detection_io import NekFramePath
from nek_post.interpolation import create_common_xz_grid, interpolate_to_grid
from nek_post.io_nek import get_nek_time, read_nek_file
from nek_post.slicing import extract_y_slice


@dataclass(frozen=True)
class ConcentrationSequence:
    """Concentration frames interpolated onto one first-snapshot grid."""

    time: NDArray[np.float64]
    file_indices: NDArray[np.int64]
    source_files: tuple[Path, ...]
    Xi: NDArray[np.float64]
    Zi: NDArray[np.float64]
    C_frames: NDArray[np.float64]
    finite_fraction: NDArray[np.float64]
    concentration_min: NDArray[np.float64]
    concentration_max: NDArray[np.float64]
    grid_metadata: Mapping[str, float | int]
    selected_y: NDArray[np.float64]
    interpolation_method: str = "linear"


def _grid_size(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer greater than or equal to 2.")
    parsed = int(value)
    if parsed < 2:
        raise ValueError(f"{name} must be greater than or equal to 2.")
    return parsed


def _frame_time(data: object, path: Path) -> float:
    raw_time = get_nek_time(data)
    try:
        time = float(raw_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has non-finite Nek time {raw_time!r}.") from exc
    if not np.isfinite(time):
        raise ValueError(f"{path} has non-finite Nek time {raw_time!r}.")
    return time


def build_concentration_sequence(
    frame_paths: tuple[NekFramePath, ...] | list[NekFramePath],
    *,
    nx: int,
    nz: int,
    slice_mode: str,
    slab_ratio: float,
    y_round_decimals: int,
    interpolation_method: str = "linear",
) -> ConcentrationSequence:
    """Read ordered Nek frames and interpolate concentration to one fixed grid."""
    frames = tuple(sorted(frame_paths, key=lambda frame: frame.index))
    if not frames:
        raise ValueError("At least one Nek frame path is required.")
    indices = [frame.index for frame in frames]
    if len(set(indices)) != len(indices):
        raise ValueError("Nek frame paths must have unique file indices.")
    nx_value = _grid_size(nx, "nx")
    nz_value = _grid_size(nz, "nz")
    if interpolation_method not in {"linear", "nearest"}:
        raise ValueError(
            "interpolation_method must be exactly 'linear' or 'nearest'."
        )

    Xi: NDArray[np.float64] | None = None
    Zi: NDArray[np.float64] | None = None
    grid_metadata: Mapping[str, float | int] | None = None
    times: list[float] = []
    concentration_frames: list[NDArray[np.float64]] = []
    finite_fractions: list[float] = []
    concentration_mins: list[float] = []
    concentration_maxes: list[float] = []
    selected_y_values: list[float] = []

    for frame in frames:
        source_path = Path(frame.path)
        try:
            data = read_nek_file(source_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read Nek frame {source_path}: {exc}") from exc
        frame_time = _frame_time(data, source_path)
        try:
            slice_data = extract_y_slice(
                data,
                slab_ratio=slab_ratio,
                mode=slice_mode,
                y_round_decimals=y_round_decimals,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract concentration slice from {source_path}: {exc}"
            ) from exc

        if Xi is None or Zi is None:
            try:
                Xi_raw, Zi_raw, _xi, _zi, metadata = create_common_xz_grid(
                    {"current_case": slice_data},
                    nx=nx_value,
                    nz=nz_value,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create fixed x-z grid from {source_path}: {exc}"
                ) from exc
            Xi = np.asarray(Xi_raw, dtype=float)
            Zi = np.asarray(Zi_raw, dtype=float)
            if Xi.ndim != 2 or Zi.ndim != 2 or Xi.shape != Zi.shape:
                raise ValueError(
                    f"Fixed grid created from {source_path} must contain matching "
                    "two-dimensional Xi and Zi arrays."
                )
            grid_metadata = MappingProxyType(dict(metadata))

        try:
            concentration = interpolate_to_grid(
                slice_data["x"],
                slice_data["z"],
                slice_data["C"],
                Xi,
                Zi,
                method=interpolation_method,
                deduplicate=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to interpolate concentration from {source_path}: {exc}"
            ) from exc
        concentration_arr = np.asarray(concentration, dtype=float)
        if concentration_arr.shape != Xi.shape:
            raise ValueError(
                f"Interpolated concentration from {source_path} has shape "
                f"{concentration_arr.shape}; expected {Xi.shape}."
            )
        finite = np.isfinite(concentration_arr)
        if not np.any(finite):
            raise ValueError(
                f"Interpolated concentration from {source_path} has no finite values."
            )

        times.append(frame_time)
        concentration_frames.append(concentration_arr)
        finite_fractions.append(float(np.count_nonzero(finite) / finite.size))
        concentration_mins.append(float(np.nanmin(concentration_arr)))
        concentration_maxes.append(float(np.nanmax(concentration_arr)))
        if slice_mode == "nearest_plane":
            selected_y_values.append(float(slice_data.get("selected_y", np.nan)))
        else:
            selected_y_values.append(float("nan"))

    time_arr = np.asarray(times, dtype=float)
    bad_steps = np.flatnonzero(np.diff(time_arr) <= 0.0)
    if bad_steps.size:
        previous = int(bad_steps[0])
        current = previous + 1
        raise ValueError(
            "Nek frame times must be strictly increasing and unique in file-index "
            f"order: {frames[previous].path} has time {time_arr[previous]:.16g}, "
            f"but {frames[current].path} has time {time_arr[current]:.16g}."
        )

    assert Xi is not None
    assert Zi is not None
    assert grid_metadata is not None
    return ConcentrationSequence(
        time=time_arr,
        file_indices=np.asarray(indices, dtype=np.int64),
        source_files=tuple(Path(frame.path) for frame in frames),
        Xi=Xi,
        Zi=Zi,
        C_frames=np.stack(concentration_frames).astype(float, copy=False),
        finite_fraction=np.asarray(finite_fractions, dtype=float),
        concentration_min=np.asarray(concentration_mins, dtype=float),
        concentration_max=np.asarray(concentration_maxes, dtype=float),
        grid_metadata=grid_metadata,
        selected_y=np.asarray(selected_y_values, dtype=float),
        interpolation_method=interpolation_method,
    )
