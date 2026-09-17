from __future__ import annotations

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from nek_post.gll import gll_nodes
from nek_post.h_refinement_mesh import analyze_mesh, analyze_mesh_refinement, read_mesh_report


def mesh(edges, *, mapping=(2, 1, 0), shape=(4, 4, 4), float32=False):
    reference = np.meshgrid(*(gll_nodes(n) for n in shape), indexing="ij")
    elements = []
    for cell in product(*(range(len(e) - 1) for e in edges)):
        coords = []
        for physical, ref in enumerate(mapping):
            lower, upper = edges[physical][cell[physical]:cell[physical] + 2]
            coord = (lower + upper) / 2 + (upper - lower) / 2 * reference[ref]
            # Reversals do not change physical interval semantics.
            coord = np.flip(coord, axis=ref) if sum(cell) % 2 else coord
            coords.append(coord)
        pos = np.stack(coords)
        if float32:
            pos = pos.astype(np.float32)
        elements.append(SimpleNamespace(pos=pos))
    np.random.default_rng(4).shuffle(elements)
    return SimpleNamespace(elem=elements)


def uniform(counts, *, shape=(4, 4, 4)):
    return mesh([np.linspace(0, 1, n + 1) for n in counts], shape=shape)


@pytest.mark.parametrize("mapping", [(2, 1, 0), (0, 2, 1), (1, 0, 2)])
def test_physical_axis_detection_stretched_mesh_and_float32(mapping):
    edges = [[-17, -16.3, -15.2], [0, .4, 1.5], [0, .1, .6, 1]]
    report = analyze_mesh(mesh(edges, mapping=mapping, float32=True))
    assert report["number_of_elements"] == report["directional_count_product"] == 12
    assert report["complete_tensor_tiling_verified"]
    assert report["physical_to_reference_axes"] == dict(zip("xyz", mapping))
    for axis, expected in zip("xyz", edges):
        direction = report["directional"][axis]
        assert direction["element_count"] == len(expected) - 1
        assert direction["polynomial_order"] == 3
        np.testing.assert_allclose(direction["element_widths"], np.diff(expected), atol=2e-6)
        assert direction["spacing"]["mean"] == pytest.approx(np.diff(expected).mean(), abs=2e-6)
        assert direction["representative_spacing"] == direction["spacing"]["median"]
    assert not report["all_elements_physically_isotropic"]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "curved"])
def test_reject_unproved_tensor_geometry(mode):
    data = uniform((2, 2, 2))
    if mode == "missing":
        data.elem.pop()
    elif mode == "duplicate":
        data.elem.append(deepcopy(data.elem[0]))
    else:
        data.elem[0].pos[0, 1, 1, 1] += 0.02
    with pytest.raises(ValueError):
        analyze_mesh(data)


@pytest.mark.parametrize("counts", [(2, 4, 8), (2, 3, 5)])
def test_isotropic_equal_and_unequal_transitions_allow_geometric_scalar_h(counts):
    reports = {label: analyze_mesh(uniform((n, n, n)))
               for label, n in zip(("a", "b", "c"), counts)}
    result = analyze_mesh_refinement(reports)
    assert result["scalar_h_justified"]
    assert list(result["scalar_h_by_case"].values()) == pytest.approx([1 / n for n in counts])
    for transition, left, right in zip(result["transitions"], counts, counts[1:]):
        assert transition["refinement_isotropic"]
        for direction in transition["directional"].values():
            assert direction["ratio_summary"]["min"] == pytest.approx(right / left)
            assert direction["matched_physical_length"] == pytest.approx(1)
            assert direction["strictly_refined_everywhere"]


def test_spatially_varying_local_ratio_rejects_scalar_despite_same_counts():
    a = analyze_mesh(uniform((2, 2, 2)))
    b = analyze_mesh(mesh([[0, .1, .5, .8, 1], np.linspace(0, 1, 5), np.linspace(0, 1, 5)]))
    result = analyze_mesh_refinement({"a": a, "b": b})
    assert not result["scalar_h_justified"]
    assert result["scalar_h_by_case"] is None
    ratios = result["transitions"][0]["directional"]["x"]
    assert ratios["directional_count_ratio"] == 2
    assert ratios["ratio_summary"]["min"] == pytest.approx(1.25)
    assert ratios["ratio_summary"]["max"] == pytest.approx(5)
    assert not ratios["constant_within_storage_tolerance"]


def test_anisotropic_refinement_does_not_hide_behind_element_count():
    a = analyze_mesh(uniform((2, 2, 2)))
    b = analyze_mesh(uniform((4, 6, 4)))
    result = analyze_mesh_refinement({"coarse": a, "fine": b})
    assert not result["scalar_h_justified"]
    assert result["transitions"][0]["single_spatially_constant_ratio"]
    assert not result["transitions"][0]["refinement_isotropic"]
    assert [result["transitions"][0]["directional"][a]["directional_count_ratio"] for a in "xyz"] == [2, 3, 2]


def test_domain_mismatch_rejected_and_order_mismatch_disqualifies_h():
    a = analyze_mesh(uniform((2, 2, 2)))
    b = analyze_mesh(mesh([np.linspace(0, 2, 5), np.linspace(0, 1, 5), np.linspace(0, 1, 5)]))
    with pytest.raises(ValueError, match="domains differ"):
        analyze_mesh_refinement({"a": a, "b": b})
    b = analyze_mesh(uniform((4, 4, 4), shape=(5, 5, 5)))
    result = analyze_mesh_refinement({"a": a, "b": b})
    assert not result["scalar_h_justified"]
    assert not result["same_polynomial_order"]


def test_more_elements_can_still_coarsen_locally():
    a = analyze_mesh(uniform((2, 2, 2)))
    b = analyze_mesh(mesh([[0, .1, .2, 1], np.linspace(0, 1, 5), np.linspace(0, 1, 5)]))
    result = analyze_mesh_refinement({"a": a, "b": b})
    x = result["transitions"][0]["directional"]["x"]
    assert x["ratio_summary"]["min"] == pytest.approx(.625)
    assert not x["no_coarsening_within_storage_tolerance"]
    assert not result["scalar_h_justified"]


def test_read_mesh_report_uses_project_reader_without_solution_variables(monkeypatch):
    import nek_post.h_refinement_mesh as module
    header = SimpleNamespace(nb_files=1, nb_elems_file=8, nb_elems=8, nb_dims=3,
                             nb_vars=(3, 3, 1, 1, 2), time=0)
    monkeypatch.setattr(module, "read_nek_header", lambda path: header)
    captured = []

    def reader(path, *, skip_vars):
        captured.append((path, skip_vars))
        return uniform((2, 2, 2))

    monkeypatch.setattr(module, "read_nek_file", reader)
    report = read_mesh_report("mesh.f00001")
    assert captured == [("mesh.f00001", ("ux", "uy", "uz", "pressure", "temperature", "s01", "s02"))]
    assert report["source_file"] == "mesh.f00001"
    assert report["source_time"] == 0
    header.nb_files = 2
    with pytest.raises(ValueError, match="Split-file"):
        read_mesh_report("mesh.f00001")
