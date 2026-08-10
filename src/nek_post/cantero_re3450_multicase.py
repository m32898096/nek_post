"""Formal Re3450 reconstructed-front comparison for N5, N7, and N9."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from nek_post.cantero_front_reconstruction import (
    CANTERO_RECONSTRUCTION_COMPARISON_COLUMNS,
    CANTERO_RECONSTRUCTION_TIMESERIES_COLUMNS,
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SLUMP_TMAX,
    DEFAULT_SLUMP_TMIN,
    DEFAULT_SMOOTH_METHOD,
    DEFAULT_SMOOTH_WINDOW,
    CanteroFrontComparison,
    CanteroFrontReconstruction,
    build_cantero_front_reconstruction_summary,
    compare_cantero_reconstructed_to_paper,
    reconstruct_cantero_mean_front,
)
from nek_post.cantero_mean_front import (
    cantero_mean_front_timeseries_path,
    read_cantero_mean_front_timeseries_csv,
)
from nek_post.front_detection_io import preflight_output_paths
from nek_post.front_io import read_digitized_paper_csv
from nek_post.front_kinematics_io import format_numeric_value


FORMAL_RE3450_CASES = ("N5", "N7", "N9")
PAPER_LABEL = "Cantero et al. (2007), Re=3450"
CASE_LABELS = MappingProxyType(
    {case: f"{case} reconstructed" for case in FORMAL_RE3450_CASES}
)

MULTICASE_SUMMARY_COLUMNS = (
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
    "n_comparison_points",
    "comparison_time_start",
    "comparison_time_end",
    "mean_signed_paper_difference",
    "mean_absolute_paper_difference",
    "rms_paper_difference",
    "max_absolute_paper_difference",
    "slump_tmin",
    "slump_tmax",
    "slumping_velocity_paper",
    "slumping_velocity_reconstructed",
    "slumping_velocity_difference",
    "slumping_velocity_relative_difference",
)


def _formal_cases(cases: Sequence[str]) -> tuple[str, str, str]:
    normalized = tuple(str(case).strip().upper() for case in cases)
    if len(normalized) != len(FORMAL_RE3450_CASES) or set(normalized) != set(
        FORMAL_RE3450_CASES
    ):
        raise ValueError("cases must contain exactly N5, N7, and N9 once each.")
    return FORMAL_RE3450_CASES


def cantero_re3450_front_csvs(
    cantero_mean_front_dir: str | Path,
) -> Mapping[str, Path]:
    """Return each formal case's own deterministic Phase-2 CSV."""
    return MappingProxyType(
        {
            case: cantero_mean_front_timeseries_path(
                cantero_mean_front_dir, case
            )
            for case in FORMAL_RE3450_CASES
        }
    )


@dataclass(frozen=True)
class CanteroRe3450MulticaseOutputPaths:
    """Every deterministic CSV and optional combined-figure output."""

    timeseries_csvs: Mapping[str, Path]
    comparison_csvs: Mapping[str, Path]
    summary_csv: Path
    linear_overlay: Path | None
    loglog_overlay: Path | None

    def __post_init__(self) -> None:
        if tuple(self.timeseries_csvs) != FORMAL_RE3450_CASES:
            raise ValueError("timeseries_csvs must be ordered N5, N7, N9.")
        if tuple(self.comparison_csvs) != FORMAL_RE3450_CASES:
            raise ValueError("comparison_csvs must be ordered N5, N7, N9.")
        if (self.linear_overlay is None) != (self.loglog_overlay is None):
            raise ValueError("Both multicase figures must be enabled or disabled together.")
        object.__setattr__(
            self, "timeseries_csvs", MappingProxyType(dict(self.timeseries_csvs))
        )
        object.__setattr__(
            self, "comparison_csvs", MappingProxyType(dict(self.comparison_csvs))
        )

    def all_paths(self) -> tuple[Path, ...]:
        csv_paths = (
            *(self.timeseries_csvs[case] for case in FORMAL_RE3450_CASES),
            *(self.comparison_csvs[case] for case in FORMAL_RE3450_CASES),
            self.summary_csv,
        )
        if self.linear_overlay is None:
            return csv_paths
        assert self.loglog_overlay is not None
        return (*csv_paths, self.linear_overlay, self.loglog_overlay)


def cantero_re3450_multicase_output_paths(
    output_dir: str | Path,
    *,
    include_plots: bool = True,
) -> CanteroRe3450MulticaseOutputPaths:
    """Build exact flat-layout outputs for the formal comparison."""
    directory = Path(output_dir)
    return CanteroRe3450MulticaseOutputPaths(
        timeseries_csvs={
            case: directory / f"{case}_cantero_reconstruction_timeseries.csv"
            for case in FORMAL_RE3450_CASES
        },
        comparison_csvs={
            case: directory / f"{case}_cantero_reconstruction_comparison.csv"
            for case in FORMAL_RE3450_CASES
        },
        summary_csv=directory / "cantero_re3450_multicase_summary.csv",
        linear_overlay=(
            directory / "cantero_re3450_multicase_linear_overlay.png"
            if include_plots
            else None
        ),
        loglog_overlay=(
            directory / "cantero_re3450_multicase_loglog_overlay.png"
            if include_plots
            else None
        ),
    )


@dataclass(frozen=True)
class OverlaySeries:
    """One paper or reconstructed series passed to a combined plot."""

    key: str
    label: str
    time: NDArray[np.float64]
    displacement: NDArray[np.float64]

    def __post_init__(self) -> None:
        time = np.array(self.time, dtype=np.float64, copy=True)
        displacement = np.array(self.displacement, dtype=np.float64, copy=True)
        if (
            time.ndim != 1
            or displacement.shape != time.shape
            or time.size == 0
            or not np.all(np.isfinite(time))
            or not np.all(np.isfinite(displacement))
        ):
            raise ValueError("Overlay time and displacement must be finite matching vectors.")
        time.setflags(write=False)
        displacement.setflags(write=False)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "displacement", displacement)


def build_cantero_re3450_overlay_series(
    paper: Mapping[str, object],
    reconstructions: Mapping[str, CanteroFrontReconstruction],
    *,
    loglog: bool,
) -> tuple[OverlaySeries, ...]:
    """Return paper plus N5/N7/N9 reconstructed series for one axis mode."""
    paper_time = np.asarray(paper["time"], dtype=np.float64)
    paper_x = np.asarray(
        paper["x"] if "x" in paper else paper["paper_x"], dtype=np.float64
    )
    raw_series = [("paper", PAPER_LABEL, paper_time, paper_x)]
    for case in FORMAL_RE3450_CASES:
        reconstruction = reconstructions[case]
        raw_series.append(
            (
                case,
                CASE_LABELS[case],
                reconstruction.time,
                reconstruction.x_reconstructed_relative,
            )
        )

    result: list[OverlaySeries] = []
    for key, label, time, displacement in raw_series:
        time_values = np.asarray(time, dtype=np.float64)
        displacement_values = np.asarray(displacement, dtype=np.float64)
        if loglog:
            mask = (
                np.isfinite(time_values)
                & np.isfinite(displacement_values)
                & (time_values > 0.0)
                & (displacement_values > 0.0)
            )
            time_values = time_values[mask]
            displacement_values = displacement_values[mask]
            if time_values.size == 0:
                raise ValueError(f"{label} has no positive finite log-log points.")
        result.append(OverlaySeries(key, label, time_values, displacement_values))
    return tuple(result)


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
                {
                    column: format_numeric_value(values[column][index])
                    for column in columns
                }
            )


def _write_multicase_csvs(
    outputs: CanteroRe3450MulticaseOutputPaths,
    reconstructions: Mapping[str, CanteroFrontReconstruction],
    comparisons: Mapping[str, CanteroFrontComparison],
    summary_rows: Sequence[Mapping[str, object]],
) -> None:
    for case in FORMAL_RE3450_CASES:
        reconstruction = reconstructions[case]
        _write_array_columns(
            outputs.timeseries_csvs[case],
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
        comparison = comparisons[case]
        _write_array_columns(
            outputs.comparison_csvs[case],
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
        writer = csv.DictWriter(handle, fieldnames=MULTICASE_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    column: format_numeric_value(row[column])
                    for column in MULTICASE_SUMMARY_COLUMNS
                }
            )


def _multicase_summary_row(
    *,
    case: str,
    front_csv: Path,
    paper_csv: Path,
    reconstruction: CanteroFrontReconstruction,
    comparison: CanteroFrontComparison,
    paper: Mapping[str, object],
    smooth_method: str,
    smooth_window: int,
    savgol_polyorder: int,
    slump_tmin: float,
    slump_tmax: float,
) -> dict[str, object]:
    base = build_cantero_front_reconstruction_summary(
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
    return {
        "case": base["case"],
        "front_source": base["front_source"],
        "paper_source": base["paper_source"],
        "smooth_method": base["smooth_method"],
        "smooth_window_requested": base["smooth_window_requested"],
        "smooth_window_effective": base["smooth_window_effective"],
        "savgol_polyorder_requested": base["savgol_polyorder_requested"],
        "savgol_polyorder_effective": base["savgol_polyorder_effective"],
        "n_front_points": base["n_front_points"],
        "time_start": base["time_start"],
        "time_end": base["time_end"],
        "x_front_start": base["x_front_start"],
        "x_front_end": base["x_front_end"],
        "x_front_relative_end": base["x_front_relative_end"],
        "x_reconstructed_start": base["x_reconstructed_start"],
        "x_reconstructed_end": base["x_reconstructed_end"],
        "x_reconstructed_relative_end": base["x_reconstructed_relative_end"],
        "mean_signed_reconstruction_difference": base[
            "mean_signed_reconstruction_difference"
        ],
        "mean_absolute_reconstruction_difference": base[
            "mean_absolute_reconstruction_difference"
        ],
        "rms_reconstruction_difference": base["rms_reconstruction_difference"],
        "max_absolute_reconstruction_difference": base[
            "max_absolute_reconstruction_difference"
        ],
        "n_comparison_points": base["n_comparison_points"],
        "comparison_time_start": base["time_min_compared"],
        "comparison_time_end": base["time_max_compared"],
        "mean_signed_paper_difference": base["mean_signed_paper_difference"],
        "mean_absolute_paper_difference": base[
            "mean_absolute_paper_difference"
        ],
        "rms_paper_difference": base["rms_paper_difference"],
        "max_absolute_paper_difference": base["max_absolute_paper_difference"],
        "slump_tmin": base["slump_tmin"],
        "slump_tmax": base["slump_tmax"],
        "slumping_velocity_paper": base["paper_slumping_velocity"],
        "slumping_velocity_reconstructed": base[
            "reconstructed_slumping_velocity"
        ],
        "slumping_velocity_difference": base["slumping_velocity_difference"],
        "slumping_velocity_relative_difference": base[
            "slumping_velocity_relative_difference"
        ],
    }


def _plot_series(
    path: Path,
    series: Sequence[OverlaySeries],
    *,
    loglog: bool,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")
    import matplotlib.pyplot as plt

    colors = {"N5": "tab:blue", "N7": "tab:orange", "N9": "tab:red"}
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    try:
        for values in series:
            if values.key == "paper":
                axis.plot(
                    values.time,
                    values.displacement,
                    linestyle="None",
                    marker="o",
                    markersize=2.8,
                    color="tab:green",
                    label=values.label,
                )
            else:
                axis.plot(
                    values.time,
                    values.displacement,
                    linewidth=1.35,
                    color=colors[values.key],
                    label=values.label,
                )
        axis.set_xscale("log" if loglog else "linear")
        axis.set_yscale("log" if loglog else "linear")
        axis.set_xlabel("Nondimensional time")
        axis.set_ylabel(r"$x_F - x_0$")
        axis.legend()
        axis.grid(True, which="both" if loglog else "major", alpha=0.25)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path, dpi=200)
    finally:
        plt.close(figure)
    return path


def write_cantero_re3450_multicase_overlays(
    outputs: CanteroRe3450MulticaseOutputPaths,
    *,
    paper: Mapping[str, object],
    reconstructions: Mapping[str, CanteroFrontReconstruction],
) -> tuple[Path, Path]:
    """Write one four-dataset linear and one four-dataset log-log overlay."""
    if outputs.linear_overlay is None or outputs.loglog_overlay is None:
        raise ValueError("Multicase plot outputs are disabled.")
    linear_series = build_cantero_re3450_overlay_series(
        paper, reconstructions, loglog=False
    )
    loglog_series = build_cantero_re3450_overlay_series(
        paper, reconstructions, loglog=True
    )
    return (
        _plot_series(outputs.linear_overlay, linear_series, loglog=False),
        _plot_series(outputs.loglog_overlay, loglog_series, loglog=True),
    )


@dataclass(frozen=True)
class CanteroRe3450MulticaseRun:
    """Completed independent reconstructions and shared-paper comparisons."""

    cases: tuple[str, str, str]
    front_csvs: Mapping[str, Path]
    paper_csv: Path
    reconstructions: Mapping[str, CanteroFrontReconstruction]
    comparisons: Mapping[str, CanteroFrontComparison]
    summary_rows: tuple[Mapping[str, object], ...]
    outputs: CanteroRe3450MulticaseOutputPaths


def run_cantero_re3450_multicase(
    *,
    front_csvs: Mapping[str, str | Path],
    paper_csv: str | Path,
    output_dir: str | Path,
    cases: Sequence[str] = FORMAL_RE3450_CASES,
    smooth_method: str = DEFAULT_SMOOTH_METHOD,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    slump_tmin: float = DEFAULT_SLUMP_TMIN,
    slump_tmax: float = DEFAULT_SLUMP_TMAX,
    overwrite: bool = False,
    no_plots: bool = False,
) -> CanteroRe3450MulticaseRun:
    """Run the formal shared-Re3450 comparison with one reconstruction per case."""
    formal_cases = _formal_cases(cases)
    resolved_front_csvs = MappingProxyType(
        {case: Path(front_csvs[case]) for case in formal_cases}
    )
    resolved_paper_csv = Path(paper_csv)
    outputs = cantero_re3450_multicase_output_paths(
        output_dir, include_plots=not no_plots
    )
    preflight_output_paths(outputs.all_paths(), overwrite)

    paper = read_digitized_paper_csv(resolved_paper_csv, require_positive=False)
    reconstructions: dict[str, CanteroFrontReconstruction] = {}
    comparisons: dict[str, CanteroFrontComparison] = {}
    summary_rows: list[Mapping[str, object]] = []
    for case in formal_cases:
        mean_front = read_cantero_mean_front_timeseries_csv(
            resolved_front_csvs[case]
        )
        reconstruction = reconstruct_cantero_mean_front(
            mean_front,
            smooth_method=smooth_method,
            smooth_window=smooth_window,
            savgol_polyorder=savgol_polyorder,
        )
        comparison = compare_cantero_reconstructed_to_paper(reconstruction, paper)
        reconstructions[case] = reconstruction
        comparisons[case] = comparison
        summary_rows.append(
            _multicase_summary_row(
                case=case,
                front_csv=resolved_front_csvs[case],
                paper_csv=resolved_paper_csv,
                reconstruction=reconstruction,
                comparison=comparison,
                paper=paper,
                smooth_method=smooth_method,
                smooth_window=smooth_window,
                savgol_polyorder=savgol_polyorder,
                slump_tmin=slump_tmin,
                slump_tmax=slump_tmax,
            )
        )

    frozen_reconstructions = MappingProxyType(dict(reconstructions))
    frozen_comparisons = MappingProxyType(dict(comparisons))
    _write_multicase_csvs(
        outputs, frozen_reconstructions, frozen_comparisons, summary_rows
    )
    if not no_plots:
        written = write_cantero_re3450_multicase_overlays(
            outputs, paper=paper, reconstructions=frozen_reconstructions
        )
        expected = (outputs.linear_overlay, outputs.loglog_overlay)
        if written != expected:
            raise RuntimeError("Multicase overlay paths did not match preflight.")
    return CanteroRe3450MulticaseRun(
        cases=formal_cases,
        front_csvs=resolved_front_csvs,
        paper_csv=resolved_paper_csv,
        reconstructions=frozen_reconstructions,
        comparisons=frozen_comparisons,
        summary_rows=tuple(summary_rows),
        outputs=outputs,
    )


__all__ = (
    "CASE_LABELS",
    "FORMAL_RE3450_CASES",
    "MULTICASE_SUMMARY_COLUMNS",
    "PAPER_LABEL",
    "CanteroRe3450MulticaseOutputPaths",
    "CanteroRe3450MulticaseRun",
    "OverlaySeries",
    "build_cantero_re3450_overlay_series",
    "cantero_re3450_front_csvs",
    "cantero_re3450_multicase_output_paths",
    "run_cantero_re3450_multicase",
    "write_cantero_re3450_multicase_overlays",
)
