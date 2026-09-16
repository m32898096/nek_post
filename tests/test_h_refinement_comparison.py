"""Time selection, all-case metrics and h-only artifact contracts."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import runpy
import sys

import numpy as np
import pytest

from nek_post.comparison import compare_sampled_grids
from nek_post.h_refinement import HRefinementStudy
from nek_post.h_refinement_slice import finish_plane_slices
from nek_post import h_refinement_comparison as h
from nek_post import h_refinement_comparison_io as output


def snapshots(*pairs):
    return tuple(h.SnapshotTime(i, t, f"GC0.f{i:05d}") for i, t in pairs)


def test_selection_uses_physical_time_not_equal_indices():
    found = [h.select_snapshot_time(c, frames, 5., max_time_error=.01)
             for c, frames in {
                 "a": snapshots((1, 0), (21, 5.002), (22, 5.25)),
                 "b": snapshots((21, 4.75), (55, 5.001), (90, 5.25)),
                 "c": snapshots((7, 4.75), (99, 5.003), (100, 5.25)),
             }.items()]
    assert [s.file_index for s in found] == [21, 55, 99]
    for s in found:
        assert s.time_error == s.actual_time - s.target_time
        assert s.absolute_time_error == abs(s.time_error)
        assert s.source_file.endswith(f"{s.file_index:05d}")


def test_time_tie_prefers_earlier_time_then_lower_index():
    result = h.select_snapshot_time("a", snapshots((8, 6), (7, 4), (2, 4)), 5, max_time_error=1)
    assert result.file_index == 2 and result.actual_time == 4


@pytest.mark.parametrize("frames,target,tolerance,match", [
    (snapshots((1, 0), (2, 1)), 2, 10, "outside"),
    (snapshots((1, 0), (2, 1)), .4, .01, "exceeds"),
    (snapshots((1, 0), (2, np.nan)), 0, .01, "nonfinite"),
    (snapshots((1, 0), (1, 1)), 0, .01, "duplicated"),
    (snapshots((1, 0)), 0, -1, "nonnegative"),
])
def test_invalid_selection(frames, target, tolerance, match):
    with pytest.raises(ValueError, match=match):
        h.select_snapshot_time("a", frames, target, max_time_error=tolerance)


def test_header_discovery(tmp_path, monkeypatch):
    for name in ("GC0.f00001", "GC0.f00015", "GC0.f00015.bak"):
        (tmp_path / name).touch()
    monkeypatch.setattr(h, "read_nek_header", lambda p: SimpleNamespace(time={"GC0.f00001": 0., "GC0.f00015": 5.}[p.name]))
    found = h.discover_snapshot_times(tmp_path, "GC0")
    assert [(s.file_index, s.physical_time) for s in found] == [(1, 0), (15, 5)]


def test_all_case_finite_and_geometry_mask_with_reference_control():
    grids = {"a": np.array([[2., 1e6, 1e6], [2., 2., 2.]]),
             "b": np.array([[3., np.nan, 3.], [3., 3., 3.]]),
             "ref": np.ones((2, 3))}
    geometry = np.ones((2, 3), bool)
    geometry[0, 2] = False
    result = compare_sampled_grids(grids, "ref", valid_mask=geometry)
    assert result.common_mask.sum() == 4
    a, b, ref = result.error_rows
    for key in ("relative_l2_error", "mean_absolute_error", "maximum_absolute_error"):
        assert a[key] == 1 and b[key] == 2 and ref[key] == 0
    assert not a["is_reference"] and a["independent_datapoint"]
    assert ref["is_reference"] and not ref["independent_datapoint"]
    assert a["valid_mask_fraction"] == 4/6


def test_zero_reference_norm_explicitly_undefined_except_control():
    result = compare_sampled_grids({"a": np.ones((2,2)), "ref": np.zeros((2,2))}, "ref")
    a, ref = result.error_rows
    assert np.isnan(a["relative_l2_error"]) and not a["relative_l2_defined"]
    assert ref["relative_l2_error"] == 0 and ref["relative_l2_defined"]
    assert a["maximum_absolute_error"] == 1


@pytest.mark.parametrize("grids,mask", [
    ({"ref": np.zeros((2,2)), "a": np.zeros((2,3))}, None),
    ({"ref": np.zeros((2,2))}, np.ones((3,2))),
    ({"ref": np.full((2,2), np.nan)}, None),
])
def test_invalid_grid_comparison(grids, mask):
    with pytest.raises(ValueError):
        compare_sampled_grids(grids, "ref", valid_mask=mask)


def inventory():
    return {"cases": [{"case": c, "number_of_elements": n} for c,n in zip(("a","b","ref"),(2,4,8))],
            "validation": dict(same_polynomial_order=True, matches_expected_order=True, same_domain_extents=True)}


def study():
    return HRefinementStudy(("a", "b", "ref"), "GC0", 7, "ref", 1e-6)


def test_reference_is_measured_not_inferred_from_name():
    data = inventory()
    assert h.validate_inventory_reference(study(), data) == dict(a=2,b=4,ref=8)
    data["cases"][0]["number_of_elements"] = 16
    with pytest.raises(ValueError, match="uniquely largest"):
        h.validate_inventory_reference(study(), data)
    data = inventory()
    data["validation"]["same_domain_extents"] = False
    with pytest.raises(ValueError, match="domain"):
        h.validate_inventory_reference(study(), data)


def slice_result():
    X, Z = np.meshgrid([0.,1.,2.], [0.,1.])
    fields = {}
    for case, offset in (("a",1.),("b",.5),("ref",0.)):
        fields[case] = {"C": 1+X+offset, "speed": 1+Z+offset, "p": X+Z+3+offset}
    masks = {c: np.ones(X.shape, bool) for c in fields}
    metadata = {"y_target": .75, "cases": {c: {} for c in fields}}
    return finish_plane_slices(X,Z,fields,masks,metadata)


def selection():
    return {c: h.SelectedTime(c,5.,5.+i*.001,21+i,i*.001,i*.001,f"/data/{c}/GC0.f{21+i:05d}")
            for i,c in enumerate(("a","b","ref"))}


def test_output_metadata_and_pressure_definition(tmp_path, monkeypatch):
    result, selected = slice_result(), selection()
    rows, masks = h.compare_h_slice(result, selected, "ref", dict(a=2,b=4,ref=8))
    assert len(rows) == 9
    assert all(r["mean_absolute_error"] == 0 for r in rows if r["field"] == "pressure_fluctuation")
    monkeypatch.setattr(output, "plot_difference", lambda *args, **kwargs: None)
    directory = tmp_path / "h_refinement"
    output.write_snapshot_artifacts(directory, result, selected, rows, masks, "ref")
    with (directory / "field_errors.csv").open() as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == 9
    assert saved[0]["target_time"] == "5.0" and saved[0]["file_index"] == "21"
    assert saved[0]["nx"] == "3" and saved[0]["nz"] == "2"
    assert saved[0]["mask_definition"] == "all_case_geometry_and_field_finite"
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["reference_case"] == "ref" and not metadata["temporal_interpolation"]
    assert metadata["selections"]["b"] == asdict(selected["b"])
    with np.load(directory / "comparison_grids.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["Xi"], result.Xi)
        assert payload["common_pressure_fluctuation_mask"].all()
    with pytest.raises(FileExistsError):
        output.write_snapshot_artifacts(directory, result, selected, rows, masks, "ref")


def test_cli_run_manifest_and_time_table(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/29_compare_h_refinement_fields.py"
    main = runpy.run_path(str(script))["main"]
    scope = main.__globals__
    monkeypatch.setattr(scope["HRefinementStudy"], "from_yaml", lambda p: study())
    paths = SimpleNamespace(case_dirs={}, case_dir=lambda c: Path(c))
    monkeypatch.setattr(scope["ProjectPaths"], "from_yaml", lambda p: paths)
    scope["inventory_study"] = lambda *a: inventory()
    scope["discover_snapshot_times"] = lambda *a: snapshots((1,0.), (21,5.), (41,10.))
    def compare(_study, _paths, selected, counts, **kwargs):
        result = slice_result()
        rows, masks = h.compare_h_slice(result, selected, "ref", counts)
        return result, rows, masks
    scope["sample_and_compare"] = compare
    scope["plot_error_history"] = lambda *a: None
    monkeypatch.setattr(output,"plot_difference",lambda *a,**k: None)
    destination = tmp_path / "results/h_refinement/test_run"
    monkeypatch.setattr(sys,"argv",[str(script),"--nx","3","--nz","2","--times","5",
                                    "--output-dir", str(destination)])
    main()
    run = json.loads((destination/"run.json").read_text())
    assert run["status"] == "complete" and run["observed_convergence_order"] is None
    assert run["nx"] == 3 and run["target_times"] == [5.]
    with (destination/"selected_times.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and {r["actual_time"] for r in rows} == {"5.0"}
    with pytest.raises(SystemExit):
        main()
