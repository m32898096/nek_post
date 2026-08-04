from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from nek_post.leading_edge_artifacts import LeadingEdgePlotData
from nek_post.paths import ProjectPaths


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "20_plot_leading_edge_evolution.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "plot_leading_edge_artifacts_script",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
plot_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(plot_script)


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(
        data_root=tmp_path / "data",
        case_dirs={"N7": tmp_path / "data" / "case_N7"},
        postproc_root=tmp_path / "postproc",
        results_root=tmp_path / "results",
        cantero_fig5a_re3450_csv=tmp_path / "paper-re3450.csv",
        cantero_fig5a_re8950_csv=tmp_path / "paper-re8950.csv",
    )


def _cases_config() -> dict[str, object]:
    return {
        "leading_edge": {
            "case": "N7",
            "reynolds_number": 3450,
            "nx": 1000,
            "z_target": 0.04,
            "threshold": 0.1,
            "y_upsample_factor": 2,
            "contour_time_spacing": 0.25,
        }
    }


def _readonly(values: object, dtype: object) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _plot_data() -> LeadingEdgePlotData:
    return LeadingEdgePlotData(
        case="N7",
        file_indices=_readonly([10, 20], np.int64),
        target_time=_readonly([0.5, 0.75], np.float64),
        actual_time=_readonly([0.5, 0.75], np.float64),
        time_error=_readonly([0.0, 0.0], np.float64),
        y=_readonly([0.0, 0.25, 0.5, 0.75], np.float64),
        x_front=_readonly(
            [[0.2, np.nan, 0.4, 0.3], [0.5, 0.6, 0.7, 0.8]],
            np.float64,
        ),
        success_mask=_readonly(
            [[True, False, True, True], [True, True, True, True]],
            np.bool_,
        ),
        crossing_count=_readonly(
            [[1, 0, 1, 1], [1, 1, 1, 1]],
            np.int64,
        ),
        threshold=0.1,
        z_target=0.04,
        nx=1000,
        native_ny=2,
        dense_ny=4,
        y_upsample_factor=2,
        y_min=0.0,
        y_max_periodic_endpoint=1.0,
        periodic_endpoint_included=False,
        target_time_spacing=0.25,
    )


def _install_success_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], LeadingEdgePlotData, ProjectPaths]:
    paths = _paths(tmp_path)
    data = _plot_data()
    calls: dict[str, object] = {
        "events": [],
        "preflight": [],
        "read": [],
        "plot": [],
    }
    monkeypatch.setattr(plot_script, "load_project_paths", lambda _p: paths)
    monkeypatch.setattr(plot_script, "load_yaml", lambda _p: _cases_config())

    def preflight(output_paths: object, overwrite: bool):
        calls["events"].append("preflight")  # type: ignore[union-attr]
        calls["preflight"].append(  # type: ignore[union-attr]
            (tuple(output_paths), overwrite)  # type: ignore[arg-type]
        )
        return tuple(output_paths)  # type: ignore[arg-type]

    def read(timeseries: Path, metadata: Path):
        calls["events"].append("read")  # type: ignore[union-attr]
        calls["read"].append((timeseries, metadata))  # type: ignore[union-attr]
        return data

    def plot(*args: object, **kwargs: object):
        calls["events"].append("plot")  # type: ignore[union-attr]
        calls["plot"].append((args, kwargs))  # type: ignore[union-attr]
        output_dir = Path(args[0])
        return [
            output_dir / "N7_leading_edge_evolution.png",
            output_dir / "N7_leading_edge_evolution.pdf",
        ]

    monkeypatch.setattr(plot_script, "preflight_output_paths", preflight)
    monkeypatch.setattr(plot_script, "read_leading_edge_artifacts", read)
    monkeypatch.setattr(plot_script, "write_leading_edge_artifact_plots", plot)
    return calls, data, paths


def test_help_succeeds_and_contains_only_artifact_plot_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        plot_script._parse_args(_paths(tmp_path), _cases_config(), ["--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    for flag in (
        "--case",
        "--timeseries-csv",
        "--metadata-csv",
        "--output-dir",
        "--reynolds-number",
        "--overwrite",
    ):
        assert flag in help_text
    for forbidden in ("--file-prefix", "--nx", "--z-target", "--all-frames"):
        assert forbidden not in help_text


def test_dynamic_input_and_output_defaults_follow_final_case(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    args = plot_script._parse_args(
        paths,
        _cases_config(),
        ["--case", "custom"],
    )

    expected_dir = paths.results_root / "leading_edge" / "CUSTOM"
    assert args.output_dir == expected_dir
    assert args.timeseries_csv == expected_dir / "CUSTOM_leading_edge_timeseries.csv"
    assert args.metadata_csv == expected_dir / "CUSTOM_leading_edge_metadata.csv"
    assert args.reynolds_number == 3450


def test_explicit_output_changes_default_inputs_and_explicit_inputs_override(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    chosen = tmp_path / "chosen"
    custom_timeseries = tmp_path / "input" / "timeseries.csv"

    args = plot_script._parse_args(
        paths,
        _cases_config(),
        [
            "--output-dir",
            str(chosen),
            "--timeseries-csv",
            str(custom_timeseries),
        ],
    )

    assert args.output_dir == chosen
    assert args.timeseries_csv == custom_timeseries
    assert args.metadata_csv == chosen / "N7_leading_edge_metadata.csv"


def test_plot_cli_preflights_both_figures_then_reads_only_csv_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, data, paths = _install_success_fakes(tmp_path, monkeypatch)
    output_dir = paths.results_root / "leading_edge" / "N7"

    plot_script.main([])

    assert calls["events"] == ["preflight", "read", "plot"]
    preflight_paths, overwrite = calls["preflight"][0]  # type: ignore[index]
    assert preflight_paths == (
        output_dir / "N7_leading_edge_evolution.png",
        output_dir / "N7_leading_edge_evolution.pdf",
    )
    assert not overwrite
    assert calls["read"] == [
        (
            output_dir / "N7_leading_edge_timeseries.csv",
            output_dir / "N7_leading_edge_metadata.csv",
        )
    ]
    plot_args, plot_kwargs = calls["plot"][0]  # type: ignore[index]
    assert plot_args == (output_dir, data)
    assert plot_kwargs == {"overwrite": False, "reynolds_number": 3450}
    output = capsys.readouterr().out
    for text in (
        "Selected frame count: 2",
        "Actual time range: 0.5 to 0.75",
        "threshold: 0.1",
        "z_target: 0.04",
        "native_ny: 2",
        "dense_ny: 4",
        "y_upsample_factor: 2",
        "N7_leading_edge_evolution.png",
        "N7_leading_edge_evolution.pdf",
    ):
        assert text in output


def test_plot_cli_has_no_nek_discovery_or_compute_entrypoints() -> None:
    for name in (
        "read_nek_file",
        "discover_nek_frame_paths",
        "build_leading_edge_evolution",
        "build_spectral_horizontal_slice_plan",
        "apply_spectral_horizontal_slice_plan",
    ):
        assert not hasattr(plot_script, name)


def test_reynolds_and_overwrite_are_passed_to_plotter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, data, _paths_value = _install_success_fakes(tmp_path, monkeypatch)

    plot_script.main(["--reynolds-number", "4000", "--overwrite"])

    _paths, overwrite = calls["preflight"][0]  # type: ignore[index]
    assert overwrite
    plot_args, plot_kwargs = calls["plot"][0]  # type: ignore[index]
    assert plot_args[1] is data
    assert plot_kwargs == {"overwrite": True, "reynolds_number": 4000.0}


def test_existing_figure_rejection_happens_before_csv_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, _data, _paths_value = _install_success_fakes(tmp_path, monkeypatch)

    def reject(_paths: object, _overwrite: bool) -> None:
        raise FileExistsError("Output exists: evolution.pdf")

    monkeypatch.setattr(plot_script, "preflight_output_paths", reject)
    with pytest.raises(SystemExit) as error:
        plot_script.main([])

    assert error.value.code == 1
    assert calls["read"] == []
    assert "evolution.pdf" in capsys.readouterr().err


def test_missing_csv_error_is_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, _data, _paths_value = _install_success_fakes(tmp_path, monkeypatch)

    def missing(timeseries: Path, _metadata: Path) -> None:
        raise FileNotFoundError(f"Leading-edge timeseries CSV not found: {timeseries}")

    monkeypatch.setattr(plot_script, "read_leading_edge_artifacts", missing)
    with pytest.raises(SystemExit) as error:
        plot_script.main([])

    assert error.value.code == 1
    assert calls["plot"] == []
    assert "timeseries CSV not found" in capsys.readouterr().err


def test_importing_plot_script_does_not_execute_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nek_post.paths as paths_module

    called: list[Path] = []

    def forbidden(path: Path) -> None:
        called.append(path)
        raise AssertionError("main executed during import")

    monkeypatch.setattr(paths_module, "load_project_paths", forbidden)
    spec = importlib.util.spec_from_file_location(
        "plot_leading_edge_artifacts_import_only",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert called == []
    assert callable(module.main)
