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
  N7: /tmp/nek-data/case_N7
  N9: /tmp/nek-data/case_N9
  N11: /tmp/nek-data/case_N11
  GC8950_N7: /tmp/nek-data/GC8950_N7
postproc_root: /tmp/nek-postproc
"""
        + results_root
        + """paper_data:
  cantero_fig5a_re3450_csv: /tmp/paper/cantero_fig5a_re3450.csv
  cantero_fig5a_re8950_csv: /tmp/paper/cantero_fig5a_re8950.csv
""",
        encoding="utf-8",
    )
    return config_path


def test_h_refinement_result_root_is_configurable_and_separate(tmp_path: Path) -> None:
    config = _write_paths_yaml(tmp_path)
    contents = config.read_text()
    config.write_text(contents.replace(
        "results_root: /tmp/nek-results\n",
        "results_root: /tmp/nek-results\n"
        "h_refinement_results_root: /tmp/nek-h-results\n",
    ))
    paths = ProjectPaths.from_yaml(config)

    assert paths.results_root == Path("/tmp/nek-results")
    assert paths.h_refinement_results_root == Path("/tmp/nek-h-results")
    assert paths.h_refinement_field_comparison_dir == Path("/tmp/nek-h-results/field_comparison")
    assert paths.h_refinement_convergence_analysis_dir == Path("/tmp/nek-h-results/convergence_analysis")
    assert paths.h_refinement_cantero_mean_front_dir == Path("/tmp/nek-h-results/cantero_mean_front")
    assert paths.h_refinement_cantero_re3450_dir == Path("/tmp/nek-h-results/cantero_re3450")
    assert paths.h_refinement_leading_edge_dir == Path("/tmp/nek-h-results/leading_edge")


def test_h_refinement_result_root_falls_back_to_data_root(tmp_path: Path) -> None:
    paths = ProjectPaths.from_yaml(_write_paths_yaml(tmp_path))
    assert paths.h_refinement_results_root == Path("/tmp/nek-data/results/h_refinement")


def test_empty_configured_h_refinement_root_is_rejected(tmp_path: Path) -> None:
    config = _write_paths_yaml(tmp_path)
    config.write_text(config.read_text().replace(
        "results_root: /tmp/nek-results\n",
        "results_root: /tmp/nek-results\nh_refinement_results_root: ''\n",
    ))
    with pytest.raises(ProjectPathsConfigError, match="h_refinement_results_root"):
        ProjectPaths.from_yaml(config)


def test_loads_valid_yaml_and_converts_strings_to_paths(tmp_path: Path) -> None:
    paths = ProjectPaths.from_yaml(_write_paths_yaml(tmp_path))

    assert paths.data_root == Path("/tmp/nek-data")
    assert paths.postproc_root == Path("/tmp/nek-postproc")
    assert paths.results_root == Path("/tmp/nek-results")
    assert all(isinstance(path, Path) for path in paths.case_dirs.values())


def test_known_and_unknown_case_lookup(tmp_path: Path) -> None:
    paths = ProjectPaths.from_yaml(_write_paths_yaml(tmp_path))

    assert paths.case_dir("N5") == Path("/tmp/nek-data/case_N5")
    assert paths.case_dir("GC3450_N5") == Path("/tmp/nek-data/case_N5")
    assert paths.case_dir("GC3450_N7") == Path("/tmp/nek-data/case_N7")
    assert paths.case_dir("GC3450_N9") == Path("/tmp/nek-data/case_N9")
    assert paths.case_dir("GC3450_N11") == Path("/tmp/nek-data/case_N11")
    assert paths.case_dir("GC8950_N7") == Path("/tmp/nek-data/GC8950_N7")
    with pytest.raises(ValueError, match="Unknown case 'missing'"):
        paths.case_dir("missing")


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
    assert paths.front_detection_cache_dir == Path(
        "/tmp/nek-postproc/front_detection_cache"
    )
    assert paths.fig5a_paper_overlay_dir == Path("/tmp/nek-results/fig5a_paper_overlay")
    assert paths.combined_xt_overlay_dir == Path("/tmp/nek-results/combined_xt_overlay")
    assert paths.cantero_equivalent_height_dir == Path(
        "/tmp/nek-results/cantero_equivalent_height"
    )
    assert paths.cantero_mean_front_dir == Path(
        "/tmp/nek-results/cantero_mean_front"
    )
    assert paths.cantero_front_reconstruction_dir == Path(
        "/tmp/nek-results/cantero_front_reconstruction"
    )
    assert paths.cantero_re3450_multicase_dir == Path(
        "/tmp/nek-results/cantero_re3450_multicase"
    )


def test_loads_cantero_figure_5a_csv_paths(tmp_path: Path) -> None:
    paths = ProjectPaths.from_yaml(_write_paths_yaml(tmp_path))

    assert paths.cantero_fig5a_re3450_csv == Path("/tmp/paper/cantero_fig5a_re3450.csv")
    assert paths.cantero_fig5a_re8950_csv == Path("/tmp/paper/cantero_fig5a_re8950.csv")


def test_repository_configuration_exposes_all_paper_and_case_datasets() -> None:
    paths = ProjectPaths.from_yaml()

    assert paths.cantero_fig5a_re3450_csv == Path(
        "/data/Nek5000_data/cantero/cantero_fig5a_3D_Re3450.csv"
    )
    assert paths.cantero_fig5a_re8950_csv == Path(
        "/data/Nek5000_data/cantero/cantero_fig5a_3D_Re8950.csv"
    )
    assert paths.case_dir("GC3450_N5") == Path("/data/Nek5000_data/case_N5")
    assert paths.case_dir("GC3450_N7") == Path("/data/Nek5000_data/case_N7")
    assert paths.case_dir("GC3450_N9") == Path("/data/Nek5000_data/case_N9")
    assert paths.case_dir("GC3450_N11") == Path("/data/Nek5000_data/case_N11")
    assert paths.case_dir("GC8950_N7") == Path(
        "/data/Nek5000_data/GC8950_N7"
    )
    assert paths.h_refinement_results_root == Path(
        "/data/Nek5000_data/results/h_refinement"
    )


def test_missing_required_key_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ProjectPathsConfigError, match="Missing required key 'results_root'"):
        ProjectPaths.from_yaml(_write_paths_yaml(tmp_path, include_results_root=False))
