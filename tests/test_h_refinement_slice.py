from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from pymech.neksuite.field import Header

from nek_post.gll import gll_nodes
from nek_post.h_refinement import HRefinementStudy
from nek_post import h_refinement_slice as hs


def mesh(xedges=(0, 1), yedges=(0, 1.5), *, n=4, curved=False):
    elements = []
    qz, qy, qx = np.meshgrid(*([gll_nodes(n)] * 3), indexing="ij")
    for x0, x1 in zip(xedges, xedges[1:]):
        for y0, y1 in zip(yedges, yedges[1:]):
            x = x0 + (qx + 1) * (x1 - x0) / 2
            y = y0 + (qy + 1) * (y1 - y0) / 2
            z = (qz + 1) / 2
            if curved:
                y = y + .04 * (1 - qy**2) * qx
            C = x**2 + 2*y + z
            elements.append(SimpleNamespace(
                pos=np.array([x, y, z]), temp=np.array([C]), scal=None,
                vel=np.array([1+x, 2*y, z]), pres=np.array([7+x+3*y-z]),
            ))
    return SimpleNamespace(elem=elements, time=0.)


def grid():
    return np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 7))


@pytest.mark.parametrize("curved", [False, True])
def test_exact_physical_plane_interpolated_polynomial(curved):
    data = mesh(curved=curved)
    X, Z = grid()
    fields, mask, diagnostics = hs.evaluate_physical_plane(data, X, Z)
    assert mask.all()
    np.testing.assert_allclose(fields["C"], X**2 + 1.5 + Z, atol=2e-10)
    np.testing.assert_allclose(fields["speed"], np.sqrt((1+X)**2 + 1.5**2 + Z**2), atol=2e-10)
    np.testing.assert_allclose(fields["p"], 9.25+X-Z, atol=2e-10)
    assert diagnostics["extrapolated_point_count"] == 0
    assert not hs.inspect_plane_geometry(data, .75)["exact_stored_plane_in_all_intersecting_elements"]


def test_different_meshes_identical_grid_and_interface_duplicates():
    cases = {"coarse": mesh(), "fine": mesh((0, .5, 1), (0, .75, 1.5))}
    geometries = {c: hs.inspect_plane_geometry(d, .75) for c, d in cases.items()}
    assert geometries["fine"]["exact_stored_plane_in_all_intersecting_elements"]
    X, Z, _ = hs.common_plane_grid(geometries, y_target=.75, nx=9, nz=7)
    fields, masks, info = {}, {}, {"cases": {}}
    for c, d in cases.items():
        fields[c], masks[c], info["cases"][c] = hs.evaluate_physical_plane(d, X, Z)
    assert info["cases"]["fine"]["ambiguous_boundary_point_count"] == X.size
    result = hs.finish_plane_slices(X, Z, fields, masks, info)
    for name in ("C", "u", "v", "w", "speed", "p", "p_prime"):
        np.testing.assert_allclose(result.fields["coarse"][name], result.fields["fine"][name], atol=1e-12)
    assert result.common_masks["all_fields"].all()
    assert abs(result.fields["fine"]["p_prime"].mean()) < 1e-12
    assert all(v == 0 for v in result.metadata["cases"]["fine"]["nan_fractions"].values())


def test_interface_discontinuity_has_single_deterministic_owner():
    data = mesh((0, .5, 1), (0, .75, 1.5))
    for i, e in enumerate(data.elem):
        e.temp[:] = i
    X, Z = grid()
    first, mask, diag = hs.evaluate_physical_plane(data, X, Z)
    second, _, _ = hs.evaluate_physical_plane(data, X, Z)
    np.testing.assert_array_equal(first["C"], second["C"])
    assert mask.all() and diag["ambiguous_boundary_point_count"] > 0
    np.testing.assert_allclose(first["C"], np.rint(first["C"]), atol=1e-12)
    assert np.min(first["C"]) >= -1e-12 and np.max(first["C"]) <= 3 + 1e-12


def test_outside_targets_and_holes_remain_nan():
    data = mesh((0, .4, .6, 1))
    del data.elem[1]  # Interior hole despite matching outer bounds.
    X, Z = np.meshgrid(np.array([-.1, .2, .5, .8, 1.1]), [0., .5, 1.])
    fields, mask, _ = hs.evaluate_physical_plane(data, X, Z)
    assert not mask[:, [0, 2, 4]].any()
    assert mask[:, [1, 3]].all()
    for a in fields.values():
        assert np.isnan(a[:, [0, 2, 4]]).all()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_domain_mismatch_rejected(axis):
    g = hs.inspect_plane_geometry(mesh(), .75)
    other = deepcopy(g)
    other["domain_bounds_xyz"][axis][1] += .1
    with pytest.raises(ValueError, match="Domain mismatch"):
        hs.common_plane_grid({"a": g, "b": other}, y_target=.75, nx=9, nz=7)


def test_small_domain_roundoff_uses_intersection():
    g = hs.inspect_plane_geometry(mesh(), .75)
    other = deepcopy(g)
    other["domain_bounds_xyz"][0][0] = 1e-7
    X, _, _ = hs.common_plane_grid({"a": g, "b": other}, y_target=.75, nx=9, nz=7)
    assert X.min() == 1e-7


def test_pressure_mean_uses_common_pressure_mask():
    X, Z = grid()
    values, mask, _ = hs.evaluate_physical_plane(mesh(), X, Z)
    fields = {"a": deepcopy(values), "b": deepcopy(values)}
    fields["b"]["p"][0, 0] = np.nan
    report = {"cases": {"a": {}, "b": {}}}
    result = hs.finish_plane_slices(X, Z, fields, {"a": mask, "b": mask}, report)
    common = result.common_masks["p"]
    assert not common[0, 0]
    for f in result.fields.values():
        assert abs(f["p_prime"][common].mean()) < 1e-12
    assert result.metadata["cases"]["b"]["nan_fractions"]["p"] == 1 / X.size


def test_permuted_reference_axes():
    data = mesh()
    for e in data.elem:
        for name in ("pos", "temp", "vel", "pres"):
            setattr(e, name, getattr(e, name).transpose(0, 3, 1, 2))
    X, Z = grid()
    fields, mask, _ = hs.evaluate_physical_plane(data, X, Z)
    assert mask.all()
    np.testing.assert_allclose(fields["C"], X**2 + 1.5 + Z, atol=1e-12)


def setup_workflow(monkeypatch, tmp_path):
    study = HRefinementStudy(("a", "b", "c"), "GC0", 3, "c", 1e-6)
    paths = SimpleNamespace(case_dir=lambda c: tmp_path / c)
    headers = {c: Header(4, (4, 4, 4), 1, 1, 0, 0, 0, 1, "XUPT") for c in study.cases}
    monkeypatch.setattr(hs, "read_nek_header", lambda p: headers[p.parent.name])
    monkeypatch.setattr(hs, "read_nek_file", lambda p, **kw: mesh())
    return study, paths, headers


def test_snapshot_workflow(monkeypatch, tmp_path):
    study, paths, _ = setup_workflow(monkeypatch, tmp_path)
    result = hs.sample_h_snapshot(study, paths, dict(a=1, b=3, c=5), nx=9, nz=7)
    assert result.Xi.shape == (7, 9)
    assert result.metadata["common_valid_fractions"]["all_fields"] == 1
    assert result.metadata["cases"]["b"]["file_index"] == 3


@pytest.mark.parametrize("fault,match", [("time", "time mismatch"), ("order", "polynomial"),
                                          ("split", "split-file"), ("variables", "lacks")])
def test_snapshot_header_rejection(monkeypatch, tmp_path, fault, match):
    study, paths, headers = setup_workflow(monkeypatch, tmp_path)
    if fault == "time":
        headers["b"].time = .01
    elif fault == "order":
        headers["b"].orders = (8, 8, 8)
    elif fault == "split":
        headers["b"].nb_files = 2
    else:
        headers["b"].nb_vars = (0, 3, 1, 1, 0)
    with pytest.raises(ValueError, match=match):
        hs.sample_h_snapshot(study, paths, dict(a=1, b=1, c=1), nx=9, nz=7)


def test_missing_case_rejected(monkeypatch, tmp_path):
    study, paths, _ = setup_workflow(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="exactly"):
        hs.sample_h_snapshot(study, paths, dict(a=1, b=1), nx=9, nz=7)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_pressure_fluctuation_matches_established_helper(nonfinite):
    from nek_post.comparison import _pressure_fluctuation

    X, Z = grid()
    values, mask, _ = hs.evaluate_physical_plane(mesh(), X, Z)
    fields = {"a": deepcopy(values), "b": deepcopy(values)}
    fields["b"]["p"][0, 0] = nonfinite
    result = hs.finish_plane_slices(
        X, Z, fields, {"a": mask, "b": mask}, {"cases": {"a": {}, "b": {}}})
    for f in result.fields.values():
        np.testing.assert_array_equal(
            f["p_prime"], _pressure_fluctuation(f["p"], result.common_masks["p"]))
    assert np.isnan(result.fields["b"]["p_prime"][0, 0])


def test_legacy_apply_still_validates_elements_without_targets():
    from nek_post.spectral_interpolation import (
        apply_spectral_slice_interpolation_plan, build_spectral_slice_interpolation_plan,
    )
    data = mesh(yedges=(0, 1, 1.5))
    plan = build_spectral_slice_interpolation_plan(data, nx=9, nz=7, y_target=.75)
    assert not np.any(plan.owner_element_index == 1)
    data.elem[1].temp = np.zeros((1, 2, 2, 2))
    with pytest.raises(ValueError, match="concentration shape mismatch"):
        apply_spectral_slice_interpolation_plan(plan, data)


def test_grid_dimensions_have_no_implicit_defaults():
    import inspect
    import runpy
    from pathlib import Path

    signature = inspect.signature(hs.sample_h_snapshot)
    for name in ("nx", "nz"):
        assert signature.parameters[name].default is inspect.Parameter.empty
    # Argument parsing must reject omitted dimensions before any raw-data I/O.
    script = Path(__file__).resolve().parents[1] / "scripts/28_extract_h_refinement_slice.py"
    return_script = runpy.run_path(str(script))
    import sys
    from unittest.mock import patch
    with patch.object(sys, "argv", [str(script), "--snapshot", "a=1,b=1,c=1",
                                    "--output-dir", "/tmp/unused_h_slice_review"]):
        with pytest.raises(SystemExit) as exc:
            return_script["main"]()
    assert exc.value.code == 2
