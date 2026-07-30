from __future__ import annotations

import importlib.util
from pathlib import Path

from nek_post.paths import ProjectPaths


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "18_n7_reconstructed_xt_fourway_overlay.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "n7_reconstructed_xt_fourway_overlay_script",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
overlay_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(overlay_script)


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(
        data_root=tmp_path / "data",
        case_dirs={
            "N7": tmp_path / "data" / "N7",
            "GC8950_N7": tmp_path / "data" / "GC8950_N7",
        },
        postproc_root=tmp_path / "postproc",
        results_root=tmp_path / "results",
        cantero_fig5a_re3450_csv=tmp_path / "paper-re3450.csv",
        cantero_fig5a_re8950_csv=tmp_path / "paper-re8950.csv",
    )


def test_cli_defaults_resolve_all_configured_inputs_and_output(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    args = overlay_script._parse_args(paths, [])

    assert args.re3450_front_csv == (
        paths.front_detection_dir
        / "N7_spectral"
        / "N7_detected_front_timeseries.csv"
    )
    assert args.re8950_front_csv == (
        paths.front_detection_dir
        / "GC8950_N7_spectral"
        / "GC8950_N7_detected_front_timeseries.csv"
    )
    assert args.re3450_paper_csv == paths.cantero_fig5a_re3450_csv
    assert args.re8950_paper_csv == paths.cantero_fig5a_re8950_csv
    assert args.output_dir == (
        paths.combined_xt_overlay_dir
        / "N7_Re3450_Re8950_reconstructed"
    )
    assert args.smooth_method == "moving_average"
    assert args.smooth_window == 11
    assert args.savgol_polyorder == 3
    assert args.slump_tmin == 3.0
    assert args.slump_tmax == 12.0
    assert not args.overwrite
    assert not args.no_plots
