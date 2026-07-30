"""Reconstruct and compare automatic N7 fronts at Re3450 and Re8950."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from nek_post.front_compare import (
    compare_front_to_paper,
    finite_mean,
    max_abs,
    mean_abs,
    rms,
    slumping_velocity_metrics,
)
from nek_post.front_io import (
    read_detected_front_timeseries_csv,
    read_digitized_paper_csv,
)
from nek_post.front_kinematics import (
    compute_kinematics,
    odd_smoothing_window,
)
from nek_post.paths import ProjectPaths


FRONT_METHOD = "automatic_spectral_element"
PAPER_ROLE = "external_published_comparison"
PAPER_DATASETS = {
    "Re3450": "Cantero_Fig5a_3D_Re3450",
    "Re8950": "Cantero_Fig5a_3D_Re8950",
}


@dataclass(frozen=True)
class ReconstructionInput:
    """One physically paired automatic-front and paper dataset."""

    key: str
    case: str
    reynolds_number: int
    front_csv: Path
    paper_csv: Path


def default_n7_reconstructed_xt_output_dir(paths: ProjectPaths) -> Path:
    """Return the established output directory for the four-way workflow."""
    return (
        paths.combined_xt_overlay_dir
        / "N7_Re3450_Re8950_reconstructed"
    )


def effective_smoothing_configuration(
    *,
    method: str,
    window: int,
    polyorder: int,
    n_points: int,
) -> tuple[int, float | int]:
    """Report the parameters actually applicable to velocity smoothing."""
    if method not in {"moving_average", "savgol"}:
        raise ValueError(
            "smooth_method must be 'moving_average' or 'savgol'."
        )
    effective_window = odd_smoothing_window(window, n_points)
    if method == "moving_average" or effective_window == 1:
        return effective_window, float("nan")

    effective_polyorder = min(
        max(0, int(polyorder)), effective_window - 1
    )
    if effective_polyorder < 1:
        return effective_window, float("nan")
    return effective_window, effective_polyorder


def reconstruct_detected_front(
    detected_front: Mapping[str, np.ndarray],
    *,
    smooth_method: str,
    smooth_window: int,
    savgol_polyorder: int,
) -> dict[str, np.ndarray]:
    """Reconstruct x-t by delegating all numerical steps to compute_kinematics."""
    time = np.asarray(detected_front["time"], dtype=np.float64)
    x_front_auto = np.asarray(
        detected_front["x_front_auto"], dtype=np.float64
    )
    kinematics = compute_kinematics(
        {"time": time, "x_front": x_front_auto},
        method=smooth_method,
        window=smooth_window,
        polyorder=savgol_polyorder,
    )
    x_reconstructed = np.asarray(
        kinematics["x_reconstructed"], dtype=np.float64
    )
    return {
        "time": time,
        "x_front_auto": x_front_auto,
        "x_detected_relative": x_front_auto - x_front_auto[0],
        "v_raw": np.asarray(kinematics["v_raw"], dtype=np.float64),
        "v_smooth": np.asarray(
            kinematics["v_smooth"], dtype=np.float64
        ),
        "x_reconstructed": x_reconstructed,
        "x_reconstructed_relative": (
            x_reconstructed - x_reconstructed[0]
        ),
        "x_reconstruction_difference": np.asarray(
            kinematics["x_reconstruction_error"], dtype=np.float64
        ),
    }


def compare_reconstructed_to_paper(
    reconstructed: Mapping[str, np.ndarray],
    paper: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compare reconstructed relative displacement at overlapping paper times."""
    comparison = compare_front_to_paper(
        dict(reconstructed),
        dict(paper),
        front_x_key="x_reconstructed_relative",
        interpolated_key="reconstructed_x_interp",
        error_key="difference",
        no_overlap_message=(
            "Reconstructed-front and paper time ranges do not overlap."
        ),
    )
    difference = comparison["difference"]
    return {
        "time": comparison["time"],
        "paper_x": comparison["paper_x"],
        "reconstructed_x_interp": comparison[
            "reconstructed_x_interp"
        ],
        "difference": difference,
        "absolute_difference": np.abs(difference),
        "relative_difference": comparison["relative_error"],
        "log_difference": comparison["log_error"],
    }


def build_reconstructed_xt_summary_row(
    *,
    reconstruction_input: ReconstructionInput,
    reconstructed: Mapping[str, np.ndarray],
    paper: Mapping[str, np.ndarray],
    comparison: Mapping[str, np.ndarray],
    smooth_method: str,
    smooth_window: int,
    savgol_polyorder: int,
    slump_tmin: float,
    slump_tmax: float,
) -> dict[str, str | int | float]:
    """Build one finite-safe summary row for a Reynolds-number case."""
    time = np.asarray(reconstructed["time"], dtype=float)
    x_front_auto = np.asarray(
        reconstructed["x_front_auto"], dtype=float
    )
    x_reconstructed = np.asarray(
        reconstructed["x_reconstructed"], dtype=float
    )
    reconstruction_difference = np.asarray(
        reconstructed["x_reconstruction_difference"], dtype=float
    )
    paper_difference = np.asarray(comparison["difference"], dtype=float)
    comparison_time = np.asarray(comparison["time"], dtype=float)
    effective_window, effective_polyorder = (
        effective_smoothing_configuration(
            method=smooth_method,
            window=smooth_window,
            polyorder=savgol_polyorder,
            n_points=time.size,
        )
    )
    slumping = slumping_velocity_metrics(
        dict(comparison),
        compared_x_key="reconstructed_x_interp",
        compared_label="reconstructed",
        tmin=slump_tmin,
        tmax=slump_tmax,
    )
    return {
        "case": reconstruction_input.case,
        "reynolds_number": reconstruction_input.reynolds_number,
        "front_source": str(reconstruction_input.front_csv),
        "front_method": FRONT_METHOD,
        "paper_dataset": PAPER_DATASETS[reconstruction_input.key],
        "paper_role": PAPER_ROLE,
        "smooth_method": smooth_method,
        "smooth_window_requested": smooth_window,
        "smooth_window_effective": effective_window,
        "savgol_polyorder_requested": savgol_polyorder,
        "savgol_polyorder_effective": effective_polyorder,
        "n_detected_points": int(time.size),
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "x_front_start": float(x_front_auto[0]),
        "x_front_end": float(x_front_auto[-1]),
        "x_reconstructed_start": float(x_reconstructed[0]),
        "x_reconstructed_end": float(x_reconstructed[-1]),
        "mean_signed_reconstruction_difference": finite_mean(
            reconstruction_difference
        ),
        "mean_absolute_reconstruction_difference": mean_abs(
            reconstruction_difference, finite_only=True
        ),
        "rms_reconstruction_difference": rms(
            reconstruction_difference, finite_only=True
        ),
        "max_absolute_reconstruction_difference": max_abs(
            reconstruction_difference, finite_only=True
        ),
        "final_reconstruction_difference": float(
            reconstruction_difference[-1]
        ),
        "n_paper_points": int(np.asarray(paper["time"]).size),
        "n_comparison_points": int(comparison_time.size),
        "time_min_compared": float(comparison_time[0]),
        "time_max_compared": float(comparison_time[-1]),
        "mean_signed_paper_difference": finite_mean(paper_difference),
        "mean_absolute_paper_difference": mean_abs(
            paper_difference, finite_only=True
        ),
        "rms_paper_difference": rms(
            paper_difference, finite_only=True
        ),
        "max_absolute_paper_difference": max_abs(
            paper_difference, finite_only=True
        ),
        "slump_tmin": slump_tmin,
        "slump_tmax": slump_tmax,
        **slumping,
    }


def run_n7_reconstructed_xt_fourway(
    *,
    re3450_front_csv: str | Path,
    re8950_front_csv: str | Path,
    re3450_paper_csv: str | Path,
    re8950_paper_csv: str | Path,
    output_dir: str | Path,
    smooth_method: str = "moving_average",
    smooth_window: int = 11,
    savgol_polyorder: int = 3,
    slump_tmin: float = 3.0,
    slump_tmax: float = 12.0,
    overwrite: bool = False,
    no_plots: bool = False,
):
    """Run both reconstructions, correct paper comparisons, and outputs."""
    if smooth_method not in {"moving_average", "savgol"}:
        raise ValueError(
            "smooth_method must be 'moving_average' or 'savgol'."
        )
    if (
        not np.isfinite(slump_tmin)
        or not np.isfinite(slump_tmax)
        or slump_tmin > slump_tmax
    ):
        raise ValueError(
            "Slumping bounds must be finite with slump_tmin <= slump_tmax."
        )

    inputs = (
        ReconstructionInput(
            "Re3450",
            "N7",
            3450,
            Path(re3450_front_csv),
            Path(re3450_paper_csv),
        ),
        ReconstructionInput(
            "Re8950",
            "GC8950_N7",
            8950,
            Path(re8950_front_csv),
            Path(re8950_paper_csv),
        ),
    )
    reconstructed_by_reynolds: dict[
        str, dict[str, np.ndarray]
    ] = {}
    papers_by_reynolds: dict[str, dict[str, np.ndarray]] = {}
    comparisons_by_reynolds: dict[
        str, dict[str, np.ndarray]
    ] = {}
    summary_rows: list[dict[str, str | int | float]] = []

    for reconstruction_input in inputs:
        detected = read_detected_front_timeseries_csv(
            reconstruction_input.front_csv
        )
        reconstructed = reconstruct_detected_front(
            detected,
            smooth_method=smooth_method,
            smooth_window=smooth_window,
            savgol_polyorder=savgol_polyorder,
        )
        paper = read_digitized_paper_csv(
            reconstruction_input.paper_csv,
            require_positive=False,
        )
        comparison = compare_reconstructed_to_paper(
            reconstructed, paper
        )
        reconstructed_by_reynolds[
            reconstruction_input.key
        ] = reconstructed
        papers_by_reynolds[reconstruction_input.key] = paper
        comparisons_by_reynolds[
            reconstruction_input.key
        ] = comparison
        summary_rows.append(
            build_reconstructed_xt_summary_row(
                reconstruction_input=reconstruction_input,
                reconstructed=reconstructed,
                paper=paper,
                comparison=comparison,
                smooth_method=smooth_method,
                smooth_window=smooth_window,
                savgol_polyorder=savgol_polyorder,
                slump_tmin=slump_tmin,
                slump_tmax=slump_tmax,
            )
        )

    from nek_post.reconstructed_xt_comparison_io import (
        preflight_reconstructed_xt_outputs,
        reconstructed_xt_output_paths,
        write_reconstructed_xt_csvs,
    )

    outputs = reconstructed_xt_output_paths(
        output_dir, include_plots=not no_plots
    )
    preflight_reconstructed_xt_outputs(outputs, overwrite)
    write_reconstructed_xt_csvs(
        outputs,
        reconstructed_by_reynolds,
        comparisons_by_reynolds,
        summary_rows,
    )

    if not no_plots:
        from nek_post.reconstructed_xt_comparison_plotting import (
            write_reconstructed_xt_plots,
        )

        written_figures = write_reconstructed_xt_plots(
            output_dir=Path(output_dir),
            reconstructed_by_reynolds=reconstructed_by_reynolds,
            papers_by_reynolds=papers_by_reynolds,
            slump_tmin=slump_tmin,
            slump_tmax=slump_tmax,
            overwrite=overwrite,
        )
        if tuple(written_figures) != outputs.figures:
            raise RuntimeError(
                "Reconstructed x-t plot paths did not match preflight."
            )

    return outputs
