"""Re=3450 reconstructed-front comparison for compatible case sets.

The historical N5/N7/N9 configuration remains the default.  Case selection,
labels, output routing, and optional numerical-reference diagnostics are
orchestration concerns; every case still uses the validated single-case
Cantero reconstruction kernels.
"""

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

REFERENCE_COMPARISON_COLUMNS = (
    "case", "reference_case", "time", "file_index",
    "x_front", "reference_x_front_interp", "front_position_difference",
    "v_raw", "reference_v_raw_interp", "raw_velocity_difference",
    "v_smooth", "reference_v_smooth_interp", "smoothed_velocity_difference",
    "x_reconstructed_relative", "reference_x_reconstructed_relative_interp",
    "reconstructed_front_difference",
)

REFERENCE_SUMMARY_COLUMNS = (
    "case", "reference_case", "is_reference", "n_comparison_points",
    "time_start", "time_end", "time_alignment",
    "mean_absolute_front_position_difference", "rms_front_position_difference",
    "max_absolute_front_position_difference",
    "mean_absolute_raw_velocity_difference", "rms_raw_velocity_difference",
    "max_absolute_raw_velocity_difference",
    "mean_absolute_smoothed_velocity_difference", "rms_smoothed_velocity_difference",
    "max_absolute_smoothed_velocity_difference",
    "mean_absolute_reconstructed_front_difference", "rms_reconstructed_front_difference",
    "max_absolute_reconstructed_front_difference",
)

INPUT_SUMMARY_COLUMNS = (
    "case", "front_source", "threshold", "reference_x", "n_input_frames",
    "n_successful_frames", "n_failed_frames", "time_start", "time_end",
    "failed_file_indices", "failed_statuses",
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


def _validated_cases(cases: Sequence[str]) -> tuple[str, ...]:
    if isinstance(cases, (str, bytes)):
        raise ValueError("cases must be a sequence of case labels.")
    normalized = tuple(str(case).strip() for case in cases)
    if not normalized or any(not case for case in normalized):
        raise ValueError("cases must contain non-empty case labels.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("cases must not contain duplicate labels.")
    if any(Path(case).name != case or case == ".." for case in normalized):
        raise ValueError("case labels must not contain path components.")
    # Preserve the original formal p-study behavior: any accepted permutation
    # was normalized to canonical N5/N7/N9 order.
    upper = tuple(case.upper() for case in normalized)
    if len(upper) == 3 and set(upper) == set(FORMAL_RE3450_CASES):
        return FORMAL_RE3450_CASES
    return normalized


def cantero_re3450_front_csvs(
    cantero_mean_front_dir: str | Path,
    cases: Sequence[str] = FORMAL_RE3450_CASES,
) -> Mapping[str, Path]:
    """Return each selected case's own deterministic Phase-2 CSV."""
    selected = _validated_cases(cases)
    return MappingProxyType(
        {
            case: cantero_mean_front_timeseries_path(
                cantero_mean_front_dir, case
            )
            for case in selected
        }
    )


@dataclass(frozen=True)
class CanteroRe3450MulticaseOutputPaths:
    """Every deterministic CSV and optional combined-figure output."""

    cases: tuple[str, ...]
    timeseries_csvs: Mapping[str, Path]
    comparison_csvs: Mapping[str, Path]
    summary_csv: Path
    linear_overlay: Path | None
    loglog_overlay: Path | None
    input_summary_csv: Path | None = None
    reference_timeseries_csv: Path | None = None
    reference_summary_csv: Path | None = None

    def __post_init__(self) -> None:
        cases = _validated_cases(self.cases)
        if tuple(self.timeseries_csvs) != cases:
            raise ValueError("timeseries_csvs must follow the selected case order.")
        if tuple(self.comparison_csvs) != cases:
            raise ValueError("comparison_csvs must follow the selected case order.")
        if (self.linear_overlay is None) != (self.loglog_overlay is None):
            raise ValueError("Both multicase figures must be enabled or disabled together.")
        object.__setattr__(
            self, "timeseries_csvs", MappingProxyType(dict(self.timeseries_csvs))
        )
        object.__setattr__(
            self, "comparison_csvs", MappingProxyType(dict(self.comparison_csvs))
        )
        object.__setattr__(self, "cases", cases)

    def all_paths(self) -> tuple[Path, ...]:
        csv_paths = (
            *(self.timeseries_csvs[case] for case in self.cases),
            *(self.comparison_csvs[case] for case in self.cases),
            self.summary_csv,
        )
        optional = tuple(path for path in (
            self.input_summary_csv, self.reference_timeseries_csv,
            self.reference_summary_csv,
        ) if path is not None)
        if self.linear_overlay is None:
            return (*csv_paths, *optional)
        assert self.loglog_overlay is not None
        return (*csv_paths, *optional, self.linear_overlay, self.loglog_overlay)


def cantero_re3450_multicase_output_paths(
    output_dir: str | Path,
    *,
    include_plots: bool = True,
    cases: Sequence[str] = FORMAL_RE3450_CASES,
    include_input_summary: bool = False,
    reference_case: str | None = None,
) -> CanteroRe3450MulticaseOutputPaths:
    """Build flat-layout outputs, preserving historical defaults exactly."""
    directory = Path(output_dir)
    selected = _validated_cases(cases)
    if reference_case is not None and reference_case not in selected:
        raise ValueError("reference_case must be one of the selected cases.")
    return CanteroRe3450MulticaseOutputPaths(
        cases=selected,
        timeseries_csvs={
            case: directory / f"{case}_cantero_reconstruction_timeseries.csv"
            for case in selected
        },
        comparison_csvs={
            case: directory / f"{case}_cantero_reconstruction_comparison.csv"
            for case in selected
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
        input_summary_csv=(directory / "cantero_re3450_input_summary.csv"
                           if include_input_summary else None),
        reference_timeseries_csv=(
            directory / f"cantero_re3450_relative_to_{reference_case}_timeseries.csv"
            if reference_case is not None else None
        ),
        reference_summary_csv=(
            directory / f"cantero_re3450_relative_to_{reference_case}_summary.csv"
            if reference_case is not None else None
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
    cases: Sequence[str] = FORMAL_RE3450_CASES,
    case_labels: Mapping[str, str] | None = None,
) -> tuple[OverlaySeries, ...]:
    """Return paper plus selected reconstructed series for one axis mode."""
    selected = _validated_cases(cases)
    labels = CASE_LABELS if case_labels is None else case_labels
    paper_time = np.asarray(paper["time"], dtype=np.float64)
    paper_x = np.asarray(
        paper["x"] if "x" in paper else paper["paper_x"], dtype=np.float64
    )
    raw_series = [("paper", PAPER_LABEL, paper_time, paper_x)]
    for case in selected:
        reconstruction = reconstructions[case]
        raw_series.append(
            (
                case,
                labels.get(case, f"{case} reconstructed"),
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
    for case in outputs.cases:
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

    colors = {"N5": "tab:blue", "N7": "tab:orange", "N9": "tab:red",
              "N7_H": "tab:blue", "N7_VH": "tab:orange", "N7_VVH": "tab:red"}
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
                    color=colors.get(values.key),
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
    cases: Sequence[str] = FORMAL_RE3450_CASES,
    case_labels: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Write linear and log-log overlays for paper plus selected cases."""
    if outputs.linear_overlay is None or outputs.loglog_overlay is None:
        raise ValueError("Multicase plot outputs are disabled.")
    linear_series = build_cantero_re3450_overlay_series(
        paper, reconstructions, loglog=False, cases=cases, case_labels=case_labels
    )
    loglog_series = build_cantero_re3450_overlay_series(
        paper, reconstructions, loglog=True, cases=cases, case_labels=case_labels
    )
    return (
        _plot_series(outputs.linear_overlay, linear_series, loglog=False),
        _plot_series(outputs.loglog_overlay, loglog_series, loglog=True),
    )


def _error_statistics(values: NDArray[np.float64], prefix: str) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        f"mean_absolute_{prefix}_difference": float(np.mean(absolute)),
        f"rms_{prefix}_difference": float(np.sqrt(np.mean(values * values))),
        f"max_absolute_{prefix}_difference": float(np.max(absolute)),
    }


def compare_cantero_reconstructions_to_reference(
    reconstructions: Mapping[str, CanteroFrontReconstruction],
    *,
    cases: Sequence[str],
    reference_case: str,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    """Compare case series at actual case times against an interpolated reference.

    The finest available numerical reference is linearly interpolated only at
    selected-case times inside its closed time range.  No extrapolation or
    temporal modification of either reconstruction is performed.
    """
    selected = _validated_cases(cases)
    if reference_case not in selected:
        raise ValueError("reference_case must be one of the selected cases.")
    if set(selected) - set(reconstructions):
        raise ValueError("reconstructions are missing one or more selected cases.")
    reference = reconstructions[reference_case]
    fields = (
        ("x_front", "front_position"),
        ("v_raw", "raw_velocity"),
        ("v_smooth", "smoothed_velocity"),
        ("x_reconstructed_relative", "reconstructed_front"),
    )
    rows: list[Mapping[str, object]] = []
    summaries: list[Mapping[str, object]] = []
    alignment = "linear reference interpolation at case times within overlap; no extrapolation"
    for case in selected:
        values = reconstructions[case]
        mask = ((values.time >= reference.time[0])
                & (values.time <= reference.time[-1]))
        if not np.any(mask):
            raise ValueError(f"{case} and {reference_case} have no overlapping times.")
        time = values.time[mask]
        file_index = values.file_index[mask]
        interpolated: dict[str, NDArray[np.float64]] = {}
        differences: dict[str, NDArray[np.float64]] = {}
        for attribute, _name in fields:
            source = np.asarray(getattr(values, attribute), dtype=np.float64)[mask]
            reference_values = np.asarray(getattr(reference, attribute), dtype=np.float64)
            interp = np.interp(time, reference.time, reference_values)
            interpolated[attribute] = interp
            differences[attribute] = source - interp
        for position, (current_time, current_index) in enumerate(
            zip(time, file_index, strict=True)
        ):
            rows.append({
                "case": case, "reference_case": reference_case,
                "time": float(current_time), "file_index": int(current_index),
                "x_front": float(values.x_front[mask][position]),
                "reference_x_front_interp": float(interpolated["x_front"][position]),
                "front_position_difference": float(differences["x_front"][position]),
                "v_raw": float(values.v_raw[mask][position]),
                "reference_v_raw_interp": float(interpolated["v_raw"][position]),
                "raw_velocity_difference": float(differences["v_raw"][position]),
                "v_smooth": float(values.v_smooth[mask][position]),
                "reference_v_smooth_interp": float(interpolated["v_smooth"][position]),
                "smoothed_velocity_difference": float(differences["v_smooth"][position]),
                "x_reconstructed_relative": float(values.x_reconstructed_relative[mask][position]),
                "reference_x_reconstructed_relative_interp": float(
                    interpolated["x_reconstructed_relative"][position]),
                "reconstructed_front_difference": float(
                    differences["x_reconstructed_relative"][position]),
            })
        summary: dict[str, object] = {
            "case": case, "reference_case": reference_case,
            "is_reference": case == reference_case,
            "n_comparison_points": int(time.size),
            "time_start": float(time[0]), "time_end": float(time[-1]),
            "time_alignment": alignment,
        }
        for attribute, name in fields:
            summary.update(_error_statistics(differences[attribute], name))
        summaries.append(summary)
    return tuple(rows), tuple(summaries)


def _write_mapping_rows(path: Path, columns: tuple[str, ...], rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_numeric_value(row[column]) for column in columns})


@dataclass(frozen=True)
class CanteroRe3450MulticaseRun:
    """Completed independent reconstructions and shared-paper comparisons."""

    cases: tuple[str, ...]
    front_csvs: Mapping[str, Path]
    paper_csv: Path
    reconstructions: Mapping[str, CanteroFrontReconstruction]
    comparisons: Mapping[str, CanteroFrontComparison]
    summary_rows: tuple[Mapping[str, object], ...]
    outputs: CanteroRe3450MulticaseOutputPaths
    input_summary_rows: tuple[Mapping[str, object], ...] = ()
    reference_timeseries_rows: tuple[Mapping[str, object], ...] = ()
    reference_summary_rows: tuple[Mapping[str, object], ...] = ()


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
    reference_case: str | None = None,
    include_input_summary: bool = False,
    case_labels: Mapping[str, str] | None = None,
) -> CanteroRe3450MulticaseRun:
    """Run the shared-Re=3450 comparison with one reconstruction per case."""
    selected_cases = _validated_cases(cases)
    if reference_case is not None and reference_case not in selected_cases:
        raise ValueError("reference_case must be one of the selected cases.")
    missing = set(selected_cases) - set(front_csvs)
    if missing:
        raise ValueError("front_csvs are missing case(s): " + ", ".join(sorted(missing)))
    resolved_front_csvs = MappingProxyType(
        {case: Path(front_csvs[case]) for case in selected_cases}
    )
    resolved_paper_csv = Path(paper_csv)
    outputs = cantero_re3450_multicase_output_paths(
        output_dir, include_plots=not no_plots, cases=selected_cases,
        include_input_summary=include_input_summary, reference_case=reference_case,
    )
    preflight_output_paths(outputs.all_paths(), overwrite)

    paper = read_digitized_paper_csv(resolved_paper_csv, require_positive=False)
    reconstructions: dict[str, CanteroFrontReconstruction] = {}
    comparisons: dict[str, CanteroFrontComparison] = {}
    summary_rows: list[Mapping[str, object]] = []
    input_summary_rows: list[Mapping[str, object]] = []
    for case in selected_cases:
        mean_front = read_cantero_mean_front_timeseries_csv(
            resolved_front_csvs[case]
        )
        if include_input_summary:
            failed_indices = tuple(int(value) for value in mean_front["failed_file_indices"])
            failed_statuses = tuple(str(value) for value in mean_front["failed_statuses"])
            input_summary_rows.append({
                "case": case, "front_source": str(resolved_front_csvs[case]),
                "threshold": mean_front["threshold"], "reference_x": mean_front["reference_x"],
                "n_input_frames": mean_front["n_input_frames"],
                "n_successful_frames": mean_front["n_successful_frames"],
                "n_failed_frames": mean_front["n_failed_frames"],
                "time_start": mean_front["input_time_start"],
                "time_end": mean_front["input_time_end"],
                "failed_file_indices": ";".join(map(str, failed_indices)),
                "failed_statuses": ";".join(failed_statuses),
            })
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
    reference_timeseries_rows: tuple[Mapping[str, object], ...] = ()
    reference_summary_rows: tuple[Mapping[str, object], ...] = ()
    if outputs.input_summary_csv is not None:
        _write_mapping_rows(outputs.input_summary_csv, INPUT_SUMMARY_COLUMNS, input_summary_rows)
    if reference_case is not None:
        reference_timeseries_rows, reference_summary_rows = (
            compare_cantero_reconstructions_to_reference(
                frozen_reconstructions, cases=selected_cases,
                reference_case=reference_case)
        )
        assert outputs.reference_timeseries_csv is not None
        assert outputs.reference_summary_csv is not None
        _write_mapping_rows(outputs.reference_timeseries_csv,
                            REFERENCE_COMPARISON_COLUMNS, reference_timeseries_rows)
        _write_mapping_rows(outputs.reference_summary_csv,
                            REFERENCE_SUMMARY_COLUMNS, reference_summary_rows)
    if not no_plots:
        written = write_cantero_re3450_multicase_overlays(
            outputs, paper=paper, reconstructions=frozen_reconstructions,
            cases=selected_cases, case_labels=case_labels,
        )
        expected = (outputs.linear_overlay, outputs.loglog_overlay)
        if written != expected:
            raise RuntimeError("Multicase overlay paths did not match preflight.")
    return CanteroRe3450MulticaseRun(
        cases=selected_cases,
        front_csvs=resolved_front_csvs,
        paper_csv=resolved_paper_csv,
        reconstructions=frozen_reconstructions,
        comparisons=frozen_comparisons,
        summary_rows=tuple(summary_rows),
        outputs=outputs,
        input_summary_rows=tuple(input_summary_rows),
        reference_timeseries_rows=reference_timeseries_rows,
        reference_summary_rows=reference_summary_rows,
    )


__all__ = (
    "CASE_LABELS",
    "FORMAL_RE3450_CASES",
    "MULTICASE_SUMMARY_COLUMNS",
    "INPUT_SUMMARY_COLUMNS",
    "REFERENCE_COMPARISON_COLUMNS",
    "REFERENCE_SUMMARY_COLUMNS",
    "PAPER_LABEL",
    "CanteroRe3450MulticaseOutputPaths",
    "CanteroRe3450MulticaseRun",
    "OverlaySeries",
    "build_cantero_re3450_overlay_series",
    "cantero_re3450_front_csvs",
    "cantero_re3450_multicase_output_paths",
    "compare_cantero_reconstructions_to_reference",
    "run_cantero_re3450_multicase",
    "write_cantero_re3450_multicase_overlays",
)
