from pathlib import Path

import pytest

from nek_post.paths import ProjectPaths, ProjectPathsConfigError


def _write_paths_yaml(tmp_path: Path, *, include_results_root: bool = True) -> Path:
    results_root = "results_root: /tmp/nek-results\n" if include_results_root else ""
    config_path = tmp_path / "paths.yaml"
    config_path.write_text(
        """data_root: /tmp/nek-data
case_dirs:
  N5: /tmp/nek-data/case_N5
  N11: /tmp/nek-data/case_N11
postproc_root: /tmp/nek-postproc
"""
        + results_root
        + """paper_data:
  cantero_fig5a_re3450_csv: /tmp/paper/cantero_fig5a_re3450.csv
""",
        encoding="utf-8",
    )
    return config_path


def test_loads_valid_yaml_and_converts_strings_to_paths(tmp_path: Path) -> None:
    paths = ProjectPaths.from_yaml(_write_paths_yaml(tmp_path))

    assert paths.data_root == Path("/tmp/nek-data")
    assert paths.postproc_root == Path("/tmp/nek-postproc")
    assert paths.results_root == Path("/tmp/nek-results")
    assert all(isinstance(path, Path) for path in paths.case_dirs.values())


def test_known_and_unknown_case_lookup(tmp_path: Path) -> None:
    paths = ProjectPaths.from_yaml(_write_paths_yaml(tmp_path))

    assert paths.case_dir("N5") == Path("/tmp/nek-data/case_N5")
    with pytest.raises(ValueError, match="Unknown case 'N7'.*N11, N5"):
        paths.case_dir("N7")


def test_derives_established_output_directories(tmp_path: Path) -> None:
    paths = ProjectPaths.from_yaml(_write_paths_yaml(tmp_path))

    assert paths.logs_dir == Path("/tmp/nek-postproc/logs")
    assert paths.slices_dir == Path("/tmp/nek-postproc/slices")
    assert paths.interpolated_dir == Path("/tmp/nek-postproc/interpolated")
    assert paths.tables_dir == Path("/tmp/nek-results/tables")
    assert paths.figures_dir == Path("/tmp/nek-results/figures")
    assert paths.reports_dir == Path("/tmp/nek-results/reports")
    assert paths.energy_budget_closure_dir == Path("/tmp/nek-results/energy_budget_closure")
    assert paths.front_kinematics_dir == Path("/tmp/nek-results/front_kinematics")
    assert paths.front_detection_dir == Path("/tmp/nek-results/front_detection")
    assert paths.fig5a_paper_overlay_dir == Path("/tmp/nek-results/fig5a_paper_overlay")
    assert paths.combined_xt_overlay_dir == Path("/tmp/nek-results/combined_xt_overlay")


def test_loads_cantero_figure_5a_csv_path(tmp_path: Path) -> None:
    paths = ProjectPaths.from_yaml(_write_paths_yaml(tmp_path))

    assert paths.cantero_fig5a_re3450_csv == Path("/tmp/paper/cantero_fig5a_re3450.csv")


def test_missing_required_key_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ProjectPathsConfigError, match="Missing required key 'results_root'"):
        ProjectPaths.from_yaml(_write_paths_yaml(tmp_path, include_results_root=False))
