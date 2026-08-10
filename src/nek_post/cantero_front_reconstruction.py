"""Phase-3 temporal reconstruction of Cantero-definition mean fronts."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from numbers import Integral
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nek_post.cantero_mean_front import read_cantero_mean_front_timeseries_csv
from nek_post.front_compare import (
    compare_front_to_paper,
    finite_mean,
    max_abs,
    mean_abs,
    rms,
    slumping_velocity_metrics,
)
from nek_post.front_detection_io import preflight_output_paths
from nek_post.front_io import read_digitized_paper_csv
from nek_post.front_kinematics import compute_kinematics, odd_smoothing_window
from nek_post.front_kinematics_io import format_numeric_value


DEFAULT_SMOOTH_METHOD = "moving_average"
DEFAULT_SMOOTH_WINDOW = 11
DEFAULT_SAVGOL_POLYORDER = 3
DEFAULT_SLUMP_TMIN = 3.0
DEFAULT_SLUMP_TMAX = 12.0

CANTERO_RECONSTRUCTION_TIMESERIES_COLUMNS = (
    "time",
    "file_index",
    "x_front",
    "x_front_relative",
    "v_raw",
    "v_smooth",
    "x_reconstructed",
    "x_reconstructed_relative",
    "x_reconstruction_difference",
)

CANTERO_RECONSTRUCTION_COMPARISON_COLUMNS = (
    "time",
    "paper_x",
    "reconstructed_x_interp",
    "difference",
    "absolute_difference",
    "relative_difference",
    "log_difference",
)

CANTERO_RECONSTRUCTION_SUMMARY_COLUMNS = (
    "case",
    "front_source",
    "paper_source",
    "smooth_method",
    "smooth_window_requested",
    "smooth_window_effective",
    "savgol_polyorder_requested",
    "savgol_polyorder_effective",
    "n_front_points",
    "time_start",
    "time_end",
    "x_front_start",
    "x_front_end",
    "x_front_relative_end",
    "x_reconstructed_start",
    "x_reconstructed_end",
    "x_reconstructed_relative_end",
    "mean_signed_reconstruction_difference",
    "mean_absolute_reconstruction_difference",
    "rms_reconstruction_difference",
    "max_absolute_reconstruction_difference",
    "final_reconstruction_difference",
    "n_paper_points",
    "n_comparison_points",
    "time_min_compared",
    "time_max_compared",
    "mean_signed_paper_difference",
    "mean_absolute_paper_difference",
    "rms_paper_difference",
    "max_absolute_paper_difference",
    "slump_tmin",
    "slump_tmax",
    "n_slumping_points",
    "paper_slumping_velocity",
    "reconstructed_slumping_velocity",
    "slumping_velocity_difference",
    "slumping_velocity_relative_difference",
)


def _readonly(values: object, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_vector(values: object, name: str, *, minimum_size: int = 2) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if result.ndim != 1 or result.size < minimum_size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with at least {minimum_size} values.")
    return result


def _file_indices(values: object, count: int) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != count or np.iscomplexobj(raw):
        raise ValueError("file_index must match the time vector.")
    try:
        result = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("file_index must be integer-valued.") from exc
    if not np.all(np.equal(raw, result)) or np.any(result < 0) or np.any(np.diff(result) <= 0):
        raise ValueError("file_index must be strictly increasing non-negative integers.")
    return result


def _smoothing_configuration(
    smooth_method: str,
    smooth_window: int,
    savgol_polyorder: int,
    *,
    n_points: int,
) -> tuple[str, int, int, float | int]:
    if smooth_method not in {"moving_average", "savgol"}:
        raise ValueError("smooth_method must be 'moving_average' or 'savgol'.")
    if isinstance(smooth_window, bool) or not isinstance(smooth_window, Integral):
        raise ValueError("smooth_window must be an integer.")
    if isinstance(savgol_polyorder, bool) or not isinstance(savgol_polyorder, Integral):
        raise ValueError("savgol_polyorder must be an integer.")
    smooth_window = int(smooth_window)
    savgol_polyorder = int(savgol_polyorder)
    effective_window = odd_smoothing_window(smooth_window, n_points)
    if smooth_method == "moving_average" or effective_window == 1:
        return smooth_method, smooth_window, savgol_polyorder, float("nan")
    effective_polyorder = min(max(0, savgol_polyorder), effective_window - 1)
    if effective_polyorder < 1:
        return smooth_method, smooth_window, savgol_polyorder, float("nan")
    return smooth_method, smooth_window, savgol_polyorder, effective_polyorder


@dataclass(frozen=True)
class CanteroFrontReconstruction:
    """Absolute and relative temporal reconstruction of a mean-front series."""

    time: NDArray[np.float64]
    file_index: NDArray[np.int64]
    x_front: NDArray[np.float64]
    x_front_relative: NDArray[np.float64]
    v_raw: NDArray[np.float64]
    v_smooth: NDArray[np.float64]
    x_reconstructed: NDArray[np.float64]
    x_reconstructed_relative: NDArray[np.float64]
    x_reconstruction_difference: NDArray[np.float64]

    def __post_init__(self) -> None:
        time = _finite_vector(self.time, "time")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time must be strictly increasing.")
        file_index = _file_indices(self.file_index, time.size)
        vectors = tuple(
            _finite_vector(values, name)
            for name, values in (
                ("x_front", self.x_front),
                ("x_front_relative", self.x_front_relative),
                ("v_raw", self.v_raw),
                ("v_smooth", self.v_smooth),
                ("x_reconstructed", self.x_reconstructed),
                ("x_reconstructed_relative", self.x_reconstructed_relative),
                ("x_reconstruction_difference", self.x_reconstruction_difference),
            )
        )
        if any(values.shape != time.shape for values in vectors):
            raise ValueError("Cantero reconstruction vectors must match time.")
        (
            x_front,
            x_front_relative,
            v_raw,
            v_smooth,
            x_reconstructed,
            x_reconstructed_relative,
            difference,
        ) = vectors
        if not np.allclose(x_front_relative, x_front - x_front[0]):
            raise ValueError("x_front_relative must use the initial absolute x_front.")
        if not np.allclose(
            x_reconstructed_relative, x_reconstructed - x_reconstructed[0]
        ):
            raise ValueError(
                "x_reconstructed_relative must use the initial reconstructed position."
            )
        if not np.allclose(difference, x_reconstructed - x_front):
            raise ValueError("x_reconstruction_difference must equal reconstructed minus front.")
        for name, values in (
            ("time", time),
            ("file_index", file_index),
            ("x_front", x_front),
            ("x_front_relative", x_front_relative),
            ("v_raw", v_raw),
            ("v_smooth", v_smooth),
            ("x_reconstructed", x_reconstructed),
            ("x_reconstructed_relative", x_reconstructed_relative),
            ("x_reconstruction_difference", difference),
        ):
            object.__setattr__(self, name, _readonly(values, values.dtype))


@dataclass(frozen=True)
class CanteroFrontComparison:
    """Reconstructed relative displacement compared with Figure-5a data."""

    time: NDArray[np.float64]
    paper_x: NDArray[np.float64]
    reconstructed_x_interp: NDArray[np.float64]
    difference: NDArray[np.float64]
    absolute_difference: NDArray[np.float64]
    relative_difference: NDArray[np.float64]
    log_difference: NDArray[np.float64]

    def __post_init__(self) -> None:
        time = _finite_vector(self.time, "comparison time")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("comparison time must be strictly increasing.")
        required = tuple(
            _finite_vector(values, name)
            for name, values in (
                ("paper_x", self.paper_x),
                ("reconstructed_x_interp", self.reconstructed_x_interp),
                ("difference", self.difference),
                ("absolute_difference", self.absolute_difference),
            )
        )
        optional = tuple(
            _readonly(values, np.float64)
            for values in (self.relative_difference, self.log_difference)
        )
        if any(values.shape != time.shape for values in (*required, *optional)):
            raise ValueError("Cantero comparison vectors must match comparison time.")
        paper_x, reconstructed, difference, absolute = required
        relative, log_difference = optional
        if not np.allclose(difference, reconstructed - paper_x):
            raise ValueError("comparison difference must equal reconstructed minus paper.")
        if not np.allclose(absolute, np.abs(difference)):
            raise ValueError("absolute_difference must equal abs(difference).")
        for name, values in (
            ("time", time),
            ("paper_x", paper_x),
            ("reconstructed_x_interp", reconstructed),
            ("difference", difference),
            ("absolute_difference", absolute),
            ("relative_difference", relative),
            ("log_difference", log_difference),
        ):
            object.__setattr__(self, name, _readonly(values, values.dtype))


@dataclass(frozen=True)
class CanteroFrontReconstructionOutputPaths:
    """Deterministic CSV and optional Figure-5a overlay paths."""

    timeseries_csv: Path
    comparison_csv: Path
    summary_csv: Path
    overlay_figure: Path | None

    def all_paths(self) -> tuple[Path, ...]:
        paths = (self.timeseries_csv, self.comparison_csv, self.summary_csv)
        return paths if self.overlay_figure is None else (*paths, self.overlay_figure)


@dataclass(frozen=True)
class CanteroFrontReconstructionRun:
    """Completed Phase-3 results, comparison, summary, and output paths."""

    reconstruction: CanteroFrontReconstruction
    comparison: CanteroFrontComparison
    summary: Mapping[str, str | int | float]
    outputs: CanteroFrontReconstructionOutputPaths


def reconstruct_cantero_mean_front(
    mean_front: Mapping[str, object],
    *,
    smooth_method: str = DEFAULT_SMOOTH_METHOD,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> CanteroFrontReconstruction:
    """Delegate mean-front temporal reconstruction to ``compute_kinematics``."""
    try:
        time = _finite_vector(mean_front["time"], "time")
        file_index = _file_indices(mean_front["file_index"], time.size)
        x_front = _finite_vector(mean_front["x_front"], "x_front")
    except KeyError as exc:
        raise ValueError(f"mean_front is missing required key {exc.args[0]!r}.") from exc
    if x_front.shape != time.shape:
        raise ValueError("x_front must match time.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing.")
    _smoothing_configuration(
        smooth_method, smooth_window, savgol_polyorder, n_points=time.size
    )
    kinematics = compute_kinematics(
        {"time": time, "x_front": x_front},
        method=smooth_method,
        window=smooth_window,
        polyorder=savgol_polyorder,
    )
    x_reconstructed = _finite_vector(
        kinematics["x_reconstructed"], "x_reconstructed"
    )
    return CanteroFrontReconstruction(
        time=time,
        file_index=file_index,
        x_front=x_front,
        x_front_relative=x_front - x_front[0],
        v_raw=_finite_vector(kinematics["v_raw"], "v_raw"),
        v_smooth=_finite_vector(kinematics["v_smooth"], "v_smooth"),
        x_reconstructed=x_reconstructed,
        x_reconstructed_relative=x_reconstructed - x_reconstructed[0],
        x_reconstruction_difference=_finite_vector(
            kinematics["x_reconstruction_error"], "x_reconstruction_error"
        ),
    )


def compare_cantero_reconstructed_to_paper(
    reconstruction: CanteroFrontReconstruction,
    paper: Mapping[str, object],
) -> CanteroFrontComparison:
    """Compare only reconstructed relative displacement with paper displacement."""
    comparison = compare_front_to_paper(
        {
            "time": reconstruction.time,
            "x_reconstructed_relative": reconstruction.x_reconstructed_relative,
        },
        dict(paper),
        front_x_key="x_reconstructed_relative",
        interpolated_key="reconstructed_x_interp",
        error_key="difference",
        no_overlap_message="Cantero reconstructed-front and paper time ranges do not overlap.",
    )
    difference = np.asarray(comparison["difference"], dtype=np.float64)
    return CanteroFrontComparison(
        time=np.asarray(comparison["time"], dtype=np.float64),
        paper_x=np.asarray(comparison["paper_x"], dtype=np.float64),
        reconstructed_x_interp=np.asarray(
            comparison["reconstructed_x_interp"], dtype=np.float64
        ),
        difference=difference,
        absolute_difference=np.abs(difference),
        relative_difference=np.asarray(comparison["relative_error"], dtype=np.float64),
        log_difference=np.asarray(comparison["log_error"], dtype=np.float64),
    )


def cantero_front_reconstruction_output_paths(
    output_dir: str | Path,
    case: str,
    *,
    include_plots: bool = True,
) -> CanteroFrontReconstructionOutputPaths:
    """Return the exact Phase-3 output paths for one case."""
    if not isinstance(case, str) or not case.strip():
        raise ValueError("case must be a non-empty string.")
    normalized_case = case.strip()
    directory = Path(output_dir) / normalized_case
    return CanteroFrontReconstructionOutputPaths(
        timeseries_csv=directory / f"{normalized_case}_cantero_front_reconstruction_timeseries.csv",
        comparison_csv=directory / f"{normalized_case}_cantero_front_reconstruction_comparison.csv",
        summary_csv=directory / f"{normalized_case}_cantero_front_reconstruction_summary.csv",
        overlay_figure=(
            directory / f"{normalized_case}_cantero_front_reconstruction_overlay.png"
            if include_plots
            else None
        ),
    )


def _effective_smoothing_values(
    reconstruction: CanteroFrontReconstruction,
    *,
    smooth_method: str,
    smooth_window: int,
    savgol_polyorder: int,
) -> tuple[int, float | int]:
    _method, _window, _polyorder, effective_polyorder = _smoothing_configuration(
        smooth_method,
        smooth_window,
        savgol_polyorder,
        n_points=reconstruction.time.size,
    )
    return odd_smoothing_window(smooth_window, reconstruction.time.size), effective_polyorder


def build_cantero_front_reconstruction_summary(
    *,
    case: str,
    front_csv: str | Path,
    paper_csv: str | Path,
    reconstruction: CanteroFrontReconstruction,
    paper: Mapping[str, object],
    comparison: CanteroFrontComparison,
    smooth_method: str,
    smooth_window: int,
    savgol_polyorder: int,
    slump_tmin: float,
    slump_tmax: float,
) -> dict[str, str | int | float]:
    """Build auditable reconstructed-only Figure-5a comparison metrics."""
    if not np.isfinite(slump_tmin) or not np.isfinite(slump_tmax) or slump_tmin > slump_tmax:
        raise ValueError("Slumping bounds must be finite with slump_tmin <= slump_tmax.")
    effective_window, effective_polyorder = _effective_smoothing_values(
        reconstruction,
        smooth_method=smooth_method,
        smooth_window=smooth_window,
        savgol_polyorder=savgol_polyorder,
    )
    slumping = slumping_velocity_metrics(
        {
            "time": comparison.time,
            "paper_x": comparison.paper_x,
            "reconstructed_x_interp": comparison.reconstructed_x_interp,
            "difference": comparison.difference,
            "relative_error": comparison.relative_difference,
        },
        compared_x_key="reconstructed_x_interp",
        compared_label="reconstructed",
        tmin=slump_tmin,
        tmax=slump_tmax,
    )
    return {
        "case": case,
        "front_source": str(front_csv),
        "paper_source": str(paper_csv),
        "smooth_method": smooth_method,
        "smooth_window_requested": smooth_window,
        "smooth_window_effective": effective_window,
        "savgol_polyorder_requested": savgol_polyorder,
        "savgol_polyorder_effective": effective_polyorder,
        "n_front_points": int(reconstruction.time.size),
        "time_start": float(reconstruction.time[0]),
        "time_end": float(reconstruction.time[-1]),
        "x_front_start": float(reconstruction.x_front[0]),
        "x_front_end": float(reconstruction.x_front[-1]),
        "x_front_relative_end": float(reconstruction.x_front_relative[-1]),
        "x_reconstructed_start": float(reconstruction.x_reconstructed[0]),
        "x_reconstructed_end": float(reconstruction.x_reconstructed[-1]),
        "x_reconstructed_relative_end": float(
            reconstruction.x_reconstructed_relative[-1]
        ),
        "mean_signed_reconstruction_difference": finite_mean(
            reconstruction.x_reconstruction_difference
        ),
        "mean_absolute_reconstruction_difference": mean_abs(
            reconstruction.x_reconstruction_difference, finite_only=True
        ),
        "rms_reconstruction_difference": rms(
            reconstruction.x_reconstruction_difference, finite_only=True
        ),
        "max_absolute_reconstruction_difference": max_abs(
            reconstruction.x_reconstruction_difference, finite_only=True
        ),
        "final_reconstruction_difference": float(
            reconstruction.x_reconstruction_difference[-1]
        ),
        "n_paper_points": int(np.asarray(paper["time"]).size),
        "n_comparison_points": int(comparison.time.size),
        "time_min_compared": float(comparison.time[0]),
        "time_max_compared": float(comparison.time[-1]),
        "mean_signed_paper_difference": finite_mean(comparison.difference),
        "mean_absolute_paper_difference": mean_abs(
            comparison.difference, finite_only=True
        ),
        "rms_paper_difference": rms(comparison.difference, finite_only=True),
        "max_absolute_paper_difference": max_abs(
            comparison.difference, finite_only=True
        ),
        "slump_tmin": slump_tmin,
        "slump_tmax": slump_tmax,
        **slumping,
    }


def _write_array_columns(
    path: Path,
    columns: tuple[str, ...],
    values: Mapping[str, NDArray[np.generic]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(np.asarray(values["time"]).size):
            writer.writerow(
                {column: format_numeric_value(values[column][index]) for column in columns}
            )


def write_cantero_front_reconstruction_csvs(
    outputs: CanteroFrontReconstructionOutputPaths,
    reconstruction: CanteroFrontReconstruction,
    comparison: CanteroFrontComparison,
    summary: Mapping[str, str | int | float],
) -> None:
    """Write Phase-3 timeseries, reconstructed-only comparison, and summary CSVs."""
    _write_array_columns(
        outputs.timeseries_csv,
        CANTERO_RECONSTRUCTION_TIMESERIES_COLUMNS,
        {
            "time": reconstruction.time,
            "file_index": reconstruction.file_index,
            "x_front": reconstruction.x_front,
            "x_front_relative": reconstruction.x_front_relative,
            "v_raw": reconstruction.v_raw,
            "v_smooth": reconstruction.v_smooth,
            "x_reconstructed": reconstruction.x_reconstructed,
            "x_reconstructed_relative": reconstruction.x_reconstructed_relative,
            "x_reconstruction_difference": reconstruction.x_reconstruction_difference,
        },
    )
    _write_array_columns(
        outputs.comparison_csv,
        CANTERO_RECONSTRUCTION_COMPARISON_COLUMNS,
        {
            "time": comparison.time,
            "paper_x": comparison.paper_x,
            "reconstructed_x_interp": comparison.reconstructed_x_interp,
            "difference": comparison.difference,
            "absolute_difference": comparison.absolute_difference,
            "relative_difference": comparison.relative_difference,
            "log_difference": comparison.log_difference,
        },
    )
    outputs.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with outputs.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANTERO_RECONSTRUCTION_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                column: format_numeric_value(summary[column])
                for column in CANTERO_RECONSTRUCTION_SUMMARY_COLUMNS
            }
        )


def write_cantero_front_reconstruction_overlay(
    output_path: str | Path,
    *,
    paper: Mapping[str, object],
    reconstruction: CanteroFrontReconstruction,
) -> Path:
    """Write the two-curve reconstructed-relative-displacement Figure-5a overlay."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")
    import matplotlib.pyplot as plt

    paper_time = _finite_vector(paper["time"], "paper time")
    paper_x = _finite_vector(
        paper["x"] if "x" in paper else paper["paper_x"], "paper displacement"
    )
    if paper_time.shape != paper_x.shape:
        raise ValueError("paper time and displacement must match.")
    path = Path(output_path)
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    try:
        axis.plot(
            paper_time,
            paper_x,
            linestyle="None",
            marker="o",
            markersize=2.5,
            color="tab:green",
            label="Cantero et al. (2007), Re=3450",
        )
        axis.plot(
            reconstruction.time,
            reconstruction.x_reconstructed_relative,
            color="tab:blue",
            linewidth=1.4,
            label="N7, reconstructed",
        )
        axis.set_xlabel("Nondimensional time")
        axis.set_ylabel(r"$x_F - x_0$")
        axis.legend()
        axis.grid(True, alpha=0.25)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path, dpi=200)
    finally:
        plt.close(figure)
    return path


def run_cantero_front_reconstruction(
    *,
    case: str,
    front_csv: str | Path,
    paper_csv: str | Path,
    output_dir: str | Path,
    smooth_method: str = DEFAULT_SMOOTH_METHOD,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    slump_tmin: float = DEFAULT_SLUMP_TMIN,
    slump_tmax: float = DEFAULT_SLUMP_TMAX,
    overwrite: bool = False,
    no_plots: bool = False,
) -> CanteroFrontReconstructionRun:
    """Run Phase 3 using successful Phase-2 rows and reconstructed-only comparison."""
    outputs = cantero_front_reconstruction_output_paths(
        output_dir, case, include_plots=not no_plots
    )
    preflight_output_paths(outputs.all_paths(), overwrite)
    mean_front = read_cantero_mean_front_timeseries_csv(front_csv)
    reconstruction = reconstruct_cantero_mean_front(
        mean_front,
        smooth_method=smooth_method,
        smooth_window=smooth_window,
        savgol_polyorder=savgol_polyorder,
    )
    paper = read_digitized_paper_csv(Path(paper_csv), require_positive=False)
    comparison = compare_cantero_reconstructed_to_paper(reconstruction, paper)
    summary = build_cantero_front_reconstruction_summary(
        case=case,
        front_csv=front_csv,
        paper_csv=paper_csv,
        reconstruction=reconstruction,
        paper=paper,
        comparison=comparison,
        smooth_method=smooth_method,
        smooth_window=smooth_window,
        savgol_polyorder=savgol_polyorder,
        slump_tmin=slump_tmin,
        slump_tmax=slump_tmax,
    )
    write_cantero_front_reconstruction_csvs(outputs, reconstruction, comparison, summary)
    if outputs.overlay_figure is not None:
        written_figure = write_cantero_front_reconstruction_overlay(
            outputs.overlay_figure, paper=paper, reconstruction=reconstruction
        )
        if written_figure != outputs.overlay_figure:
            raise RuntimeError("Cantero reconstruction overlay path did not match preflight.")
    return CanteroFrontReconstructionRun(reconstruction, comparison, summary, outputs)


__all__ = (
    "CANTERO_RECONSTRUCTION_COMPARISON_COLUMNS",
    "CANTERO_RECONSTRUCTION_SUMMARY_COLUMNS",
    "CANTERO_RECONSTRUCTION_TIMESERIES_COLUMNS",
    "DEFAULT_SAVGOL_POLYORDER",
    "DEFAULT_SLUMP_TMAX",
    "DEFAULT_SLUMP_TMIN",
    "DEFAULT_SMOOTH_METHOD",
    "DEFAULT_SMOOTH_WINDOW",
    "CanteroFrontComparison",
    "CanteroFrontReconstruction",
    "CanteroFrontReconstructionOutputPaths",
    "CanteroFrontReconstructionRun",
    "build_cantero_front_reconstruction_summary",
    "cantero_front_reconstruction_output_paths",
    "compare_cantero_reconstructed_to_paper",
    "reconstruct_cantero_mean_front",
    "run_cantero_front_reconstruction",
    "write_cantero_front_reconstruction_csvs",
    "write_cantero_front_reconstruction_overlay",
)
