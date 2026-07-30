"""Output paths and CSV writing for the N7 reconstructed x-t comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nek_post.front_detection_io import preflight_output_paths
from nek_post.front_kinematics_io import format_numeric_value


RECONSTRUCTED_TIMESERIES_COLUMNS = (
    "time",
    "x_front_auto",
    "x_detected_relative",
    "v_raw",
    "v_smooth",
    "x_reconstructed",
    "x_reconstructed_relative",
    "x_reconstruction_difference",
)

PAPER_COMPARISON_COLUMNS = (
    "time",
    "paper_x",
    "reconstructed_x_interp",
    "difference",
    "absolute_difference",
    "relative_difference",
    "log_difference",
)

SUMMARY_COLUMNS = (
    "case",
    "reynolds_number",
    "front_source",
    "front_method",
    "paper_dataset",
    "paper_role",
    "smooth_method",
    "smooth_window_requested",
    "smooth_window_effective",
    "savgol_polyorder_requested",
    "savgol_polyorder_effective",
    "n_detected_points",
    "time_start",
    "time_end",
    "x_front_start",
    "x_front_end",
    "x_reconstructed_start",
    "x_reconstructed_end",
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


@dataclass(frozen=True)
class ReconstructedXTOutputPaths:
    """All deterministic outputs for the two-Reynolds-number workflow."""

    re3450_timeseries_csv: Path
    re8950_timeseries_csv: Path
    re3450_comparison_csv: Path
    re8950_comparison_csv: Path
    summary_csv: Path
    figures: tuple[Path, ...]

    def all_paths(self) -> tuple[Path, ...]:
        """Return every enabled output in preflight order."""
        return (
            self.re3450_timeseries_csv,
            self.re8950_timeseries_csv,
            self.re3450_comparison_csv,
            self.re8950_comparison_csv,
            self.summary_csv,
            *self.figures,
        )


def reconstructed_xt_output_paths(
    output_dir: str | Path,
    *,
    include_plots: bool = True,
) -> ReconstructedXTOutputPaths:
    """Build exact CSV and optional figure paths."""
    directory = Path(output_dir)
    figures = (
        directory
        / "Re3450_N7_reconstructed_vs_Cantero_Re3450.png",
        directory
        / "Re8950_N7_reconstructed_vs_Cantero_Re8950.png",
        directory
        / "N7_Re3450_Re8950_reconstructed_fourway_overlay.png",
    )
    return ReconstructedXTOutputPaths(
        re3450_timeseries_csv=(
            directory / "Re3450_N7_detected_front_reconstructed_xt.csv"
        ),
        re8950_timeseries_csv=(
            directory / "Re8950_N7_detected_front_reconstructed_xt.csv"
        ),
        re3450_comparison_csv=(
            directory
            / "Re3450_N7_reconstructed_vs_Cantero_Re3450.csv"
        ),
        re8950_comparison_csv=(
            directory
            / "Re8950_N7_reconstructed_vs_Cantero_Re8950.csv"
        ),
        summary_csv=(
            directory / "N7_Re3450_Re8950_reconstructed_xt_summary.csv"
        ),
        figures=figures if include_plots else (),
    )


def preflight_reconstructed_xt_outputs(
    outputs: ReconstructedXTOutputPaths,
    overwrite: bool,
) -> None:
    """Protect every enabled output before any file is written."""
    preflight_output_paths(outputs.all_paths(), overwrite)


def _write_array_columns(
    path: Path,
    columns: tuple[str, ...],
    data: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(np.asarray(data["time"]).size):
            writer.writerow(
                {
                    column: format_numeric_value(data[column][index])
                    for column in columns
                }
            )


def write_reconstructed_xt_csvs(
    outputs: ReconstructedXTOutputPaths,
    reconstructed_by_reynolds: Mapping[str, Mapping[str, np.ndarray]],
    comparisons_by_reynolds: Mapping[str, Mapping[str, np.ndarray]],
    summary_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write both reconstructed series, both comparisons, and one summary."""
    _write_array_columns(
        outputs.re3450_timeseries_csv,
        RECONSTRUCTED_TIMESERIES_COLUMNS,
        reconstructed_by_reynolds["Re3450"],
    )
    _write_array_columns(
        outputs.re8950_timeseries_csv,
        RECONSTRUCTED_TIMESERIES_COLUMNS,
        reconstructed_by_reynolds["Re8950"],
    )
    _write_array_columns(
        outputs.re3450_comparison_csv,
        PAPER_COMPARISON_COLUMNS,
        comparisons_by_reynolds["Re3450"],
    )
    _write_array_columns(
        outputs.re8950_comparison_csv,
        PAPER_COMPARISON_COLUMNS,
        comparisons_by_reynolds["Re8950"],
    )

    outputs.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with outputs.summary_csv.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    column: format_numeric_value(row[column])
                    for column in SUMMARY_COLUMNS
                }
            )
