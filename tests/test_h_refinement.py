"""Synthetic inventory and p-study isolation regression tests."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pymech.neksuite.field import Header

from nek_post import h_refinement as inv
from nek_post.config import load_yaml
from nek_post.paths import ProjectPaths

ROOT = Path(__file__).resolve().parents[1]


def config():
    return dict(cases=["a", "b", "c"], file_prefix="GC0", expected_polynomial_order=7,
                provisional_reference_case="c", domain_atol=1e-6)


def install_case(tmp_path, monkeypatch, *, orders=(8, 8, 8), times=(0, .25, .5), indices=(1, 2, 3)):
    headers = {}
    for index, time in zip(indices, times):
        path = tmp_path / f"GC0.f{index:05d}"
        path.touch()
        headers[path] = Header(4, orders, 2, 2, time, index, 0, 1, "XUPTS02")
    (tmp_path / "GC0.f00001.bak").touch()
    (tmp_path / "other.f00004").touch()
    shape = (3, *orders[::-1])
    first = np.zeros(shape)
    second = np.ones(shape)
    second[0] *= 12
    second[1] *= 1.5
    calls = []

    def reader(path, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace(elem=[SimpleNamespace(pos=first), SimpleNamespace(pos=second)])

    monkeypatch.setattr(inv, "read_nek_header", headers.__getitem__)
    monkeypatch.setattr(inv, "read_nek_file", reader)
    return headers, calls


def test_inventory(tmp_path, monkeypatch):
    _, calls = install_case(tmp_path, monkeypatch)
    result = inv.inventory_case("a", tmp_path, file_prefix="GC0")
    assert result["field_file_count"] == 3
    assert result["first_file_index"] == 1 and result["last_file_index"] == 3
    assert result["physical_time_range"] == [0, .5]
    assert result["output_time_spacing"]["median"] == .25
    assert result["output_time_spacing"]["uniform"]
    assert result["element_local_scalar_shape_zyx"] == [8, 8, 8]
    assert result["gll_nodes_per_element"] == 512
    assert result["polynomial_order"] == 7
    assert result["number_of_elements"] == 2
    assert result["domain_extents"] == dict(x=[0, 12], y=[0, 1.5], z=[0, 1])
    assert result["solution_variables"] == ["passive_scalars", "pressure", "temperature", "velocity"]
    assert len(calls) == 1 and "s02" in calls[0][1]["skip_vars"]


@pytest.mark.parametrize("times,indices,spacing,warnings", [
    ((0,), (8,), None, False),
    ((0, .2, .5), (1, 2, 3), .25, False),
    ((0, .5, .25), (1, 2, 4), None, True),
    ((0, 0, .5), (1, 2, 3), None, True),
])
def test_spacing(tmp_path, monkeypatch, times, indices, spacing, warnings):
    install_case(tmp_path, monkeypatch, times=times, indices=indices)
    result = inv.inventory_case("a", tmp_path, file_prefix="GC0")
    if spacing is None:
        assert result["output_time_spacing"] is None
    else:
        assert result["output_time_spacing"]["median"] == spacing
        assert not result["output_time_spacing"]["uniform"]
    assert bool(result["warnings"]) == warnings


@pytest.mark.parametrize("orders,order", [((8, 8, 1), 7), ((8, 6, 8), None)])
def test_dimension_and_anisotropic_order(tmp_path, monkeypatch, orders, order):
    install_case(tmp_path, monkeypatch, orders=orders)
    result = inv.inventory_case("a", tmp_path, file_prefix="GC0")
    assert result["polynomial_order"] == order
    assert result["element_local_scalar_shape_zyx"] == list(orders[::-1])


@pytest.mark.parametrize("change,match", [
    ({"nb_files": 2}, "split-file"), ({"orders": (6, 6, 6)}, "changes"),
    ({"nb_elems": 3, "nb_elems_file": 3}, "changes"), ({"time": np.nan}, "nonfinite"),
])
def test_invalid_headers(tmp_path, monkeypatch, change, match):
    headers, _ = install_case(tmp_path, monkeypatch)
    header = list(headers.values())[-1]
    for key, value in change.items():
        setattr(header, key, value)
    with pytest.raises(ValueError, match=match):
        inv.inventory_case("a", tmp_path, file_prefix="GC0")


def test_missing_coordinates(tmp_path, monkeypatch):
    headers, _ = install_case(tmp_path, monkeypatch)
    for header in headers.values():
        header.nb_vars = (0, 3, 1, 1, 0)
    with pytest.raises(ValueError, match="no coordinate"):
        inv.inventory_case("a", tmp_path, file_prefix="GC0")


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        inv.inventory_case("a", tmp_path, file_prefix="GC0")


@pytest.mark.parametrize("key,value", [
    ("cases", ["a", "a"]), ("cases", "abc"), ("cases", ["a"]),
    ("file_prefix", ""), ("file_prefix", "../GC0"),
    ("expected_polynomial_order", True), ("expected_polynomial_order", 0),
    ("provisional_reference_case", "missing"), ("domain_atol", -1), ("domain_atol", float("nan")),
])
def test_invalid_config(key, value):
    values = config()
    values[key] = value
    with pytest.raises(ValueError):
        inv.HRefinementStudy.from_mapping(values)


@pytest.mark.parametrize("fault", [None, "order", "bounds", "count", "reference", "anisotropic"])
def test_study_validation(monkeypatch, fault):
    values = config()
    if fault == "reference":
        values["provisional_reference_case"] = "a"
    reports = {c: dict(polynomial_order=7, dimension=3, number_of_elements=n,
                      domain_extents=dict(x=[0, 12], y=[0, 1.5], z=[0, 1]))
               for c, n in zip(values["cases"], (2, 4, 8))}
    if fault == "order":
        reports["b"]["polynomial_order"] = 5
    if fault == "anisotropic":
        reports["b"]["polynomial_order"] = None
    if fault == "bounds":
        reports["b"]["domain_extents"]["x"] = [0, 13]
    if fault == "count":
        reports["b"]["number_of_elements"] = 8
    monkeypatch.setattr(inv, "inventory_case", lambda c, d, **kw: reports[c])
    paths = SimpleNamespace(case_dir=lambda c: Path(c))
    validation = inv.inventory_study(inv.HRefinementStudy.from_mapping(values), paths)["validation"]
    assert validation["consistent_with_h_refinement"] is (fault is None)


def test_repository_config_isolation():
    cases = load_yaml(ROOT / "config/cases.yaml")
    assert cases["orders"] == {"N5": 5, "N7": 7, "N9": 9, "N11": 11}
    assert cases["reference_case"] == "N11"
    study = inv.HRefinementStudy.from_yaml(ROOT / "config/h_refinement.yaml")
    paths = ProjectPaths.from_yaml(ROOT / "config/paths.yaml")
    assert set(study.cases).isdisjoint(cases["orders"])
    for label in study.cases:
        assert paths.case_dir(label) == Path("/data/Nek5000_data") / f"case_{label}"
    for comparison in cases["comparison_sets"].values():
        assert set(comparison["case_indices"]) == set(cases["orders"])
