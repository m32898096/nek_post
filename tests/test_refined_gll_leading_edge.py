from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import nek_post.refined_gll_leading_edge as adapter
from nek_post.gll import gll_nodes
from nek_post.leading_edge_methods import LeadingEdgeExtractionResult, extract_leading_edge
from nek_post.refined_gll_horizontal_slice import (
    RefinedGLLHorizontalSlice,
    apply_refined_gll_horizontal_slice_plan,
    build_refined_gll_horizontal_slice_plan,
)


METHODS = ("rightmost-crossing", "moore-boundary")


@pytest.fixture
def context() -> tuple:
    q0, q1, q2 = np.meshgrid(*(gll_nodes(8),) * 3, indexing="ij")
    x, y, z = 0.5 + 1.5*q2, 0.5*(q1+1.0), 0.5*(q0+1.0)
    data = SimpleNamespace(elem=[SimpleNamespace(pos=[x, y, z], temp=[1.0-x])])
    plan = build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)
    plane = apply_refined_gll_horizontal_slice_plan(data, plan)
    return data, plan, plane


def _plane(
    template: RefinedGLLHorizontalSlice, x: object, row: object,
) -> RefinedGLLHorizontalSlice:
    x_values = np.asarray(x, dtype=np.float64)
    concentration = np.broadcast_to(row, (template.y.size, x_values.size)).copy()
    return replace(template, x=x_values, concentration=concentration)


def _direct(
    plane: RefinedGLLHorizontalSlice, method: object, **options: object,
) -> LeadingEdgeExtractionResult:
    return extract_leading_edge(
        plane.x, plane.y, plane.concentration, method=method,
        periodic_y=plane.periodic_y, y_period=plane.y_period, **options,
    )


def _assert_equal(
    actual: LeadingEdgeExtractionResult, expected: LeadingEdgeExtractionResult,
) -> None:
    assert type(actual) is LeadingEdgeExtractionResult
    assert actual.method == expected.method
    assert actual.threshold == expected.threshold
    for name in ("y", "x_front", "success_mask", "crossing_count"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("method", METHODS)
def test_full_adapter_extracts_analytic_front_without_mutating_data_or_plan(
    context: tuple, method: str,
) -> None:
    data, plan, plane = context
    before_plan = {
        name: value.copy() for name, value in vars(plan).items()
        if isinstance(value, np.ndarray)
    }
    before_data = [value.copy() for value in (*data.elem[0].pos, data.elem[0].temp[0])]
    result = adapter.extract_refined_gll_leading_edge_frame(
        data, plan=plan, threshold=0.2, extraction_method=method, x_min=0.0
    )
    _assert_equal(result, _direct(plane, method, threshold=0.2, x_min=0.0))
    np.testing.assert_allclose(result.x_front, 0.8, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(result.y, plane.y)
    assert result.x_front.shape == (plane.y.size,)
    assert np.all(result.success_mask)
    for name, before in before_plan.items():
        np.testing.assert_array_equal(getattr(plan, name), before)
    for actual, before in zip((*data.elem[0].pos, data.elem[0].temp[0]), before_data, strict=True):
        np.testing.assert_array_equal(actual, before)


@pytest.mark.parametrize("method", METHODS)
def test_nonuniform_physical_coordinates_and_period_are_forwarded_unchanged(
    context: tuple, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    data, plan, template = context
    x = np.array([-2.0, -0.1, 0.0, 0.02, 0.05, 0.4, 1.7, 4.0, 9.0])
    plane = _plane(template, x, 0.1 + 1.3-x)
    originals = tuple(a.copy() for a in (plane.x, plane.y, plane.concentration))
    calls = []
    results = []

    def apply(supplied_data: object, supplied_plan: object) -> RefinedGLLHorizontalSlice:
        assert supplied_data is data
        assert supplied_plan is plan
        return plane

    def dispatch(*args: object, **kwargs: object) -> LeadingEdgeExtractionResult:
        calls.append((args, kwargs))
        results.append(extract_leading_edge(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(adapter, "apply_refined_gll_horizontal_slice_plan", apply)
    monkeypatch.setattr(adapter, "extract_leading_edge", dispatch)
    result = adapter.extract_refined_gll_leading_edge_frame(
        data, plan=plan, threshold=0.1, extraction_method=method, x_min=0.0
    )
    assert result is results[0]
    assert len(calls) == 1
    args, kwargs = calls[0]
    for actual, original in zip(args, (plane.x, plane.y, plane.concentration), strict=True):
        assert actual is original
    assert kwargs == dict(threshold=0.1, method=method, periodic_y=True, y_period=1.0, x_min=0.0)
    np.testing.assert_allclose(result.x_front, 1.3, rtol=0.0, atol=3e-16)
    np.testing.assert_array_equal(result.y, plane.y)
    assert result.y.size == 9
    assert result.y[-1] < result.y[0] + plane.y_period
    for actual, before in zip((plane.x, plane.y, plane.concentration), originals, strict=True):
        np.testing.assert_array_equal(actual, before)


def test_rightmost_selects_last_of_multiple_physical_intersections(
    context: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, plan, template = context
    plane = _plane(template, [0.01, 0.03, 0.2, 1.5, 4.0, 10.0], [0., 1., 0., 1., 0., 0.])
    monkeypatch.setattr(adapter, "apply_refined_gll_horizontal_slice_plan", lambda *_: plane)
    result = adapter.extract_refined_gll_leading_edge_frame(
        data, plan=plan, threshold=0.5, extraction_method="rightmost-crossing"
    )
    np.testing.assert_array_equal(result.crossing_count, 4)
    np.testing.assert_array_equal(result.x_front, 2.75)


def test_moore_periodic_topology_matches_dispatch_on_nonuniform_grid(
    context: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, plan, template = context
    x = np.array([0.01, 0.03, 0.2, 1.5, 4.0, 10.0])
    plane = _plane(template, x, [1., 1., 1., 0., 0., 0.])
    field = plane.concentration.copy()
    # A protrusion connects across the periodic seam between the last/first row.
    field[[0, -1], 3] = 1.0
    plane = replace(plane, concentration=field)
    monkeypatch.setattr(adapter, "apply_refined_gll_horizontal_slice_plan", lambda *_: plane)
    result = adapter.extract_refined_gll_leading_edge_frame(
        data, plan=plan, threshold=0.5, extraction_method="moore-boundary"
    )
    _assert_equal(result, _direct(plane, "moore-boundary", threshold=0.5))
    assert np.all(result.success_mask)
    np.testing.assert_array_equal(result.x_front[[0, -1]], 2.75)
    np.testing.assert_allclose(result.x_front[1:-1], 0.85, rtol=0.0, atol=2e-16)
    # Relabeling coordinates preserves Moore's grid neighbourhood semantics.
    uniform = replace(plane, x=np.arange(x.size), y=np.arange(plane.y.size)/plane.y.size)
    reference = _direct(uniform, "moore-boundary", threshold=0.5)
    np.testing.assert_array_equal(result.success_mask, reference.success_mask)
    np.testing.assert_array_equal(result.crossing_count, reference.crossing_count)


@pytest.mark.parametrize("method", METHODS)
def test_x_min_zero_excludes_the_exact_zero_column_before_extraction(
    context: tuple, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    data, plan, template = context
    x = np.array([-2., -0.1, 0., 0.02, 0.4, 1.7, 4.])
    plane = _plane(template, x, 0.1-x)
    monkeypatch.setattr(adapter, "apply_refined_gll_horizontal_slice_plan", lambda *_: plane)
    unbounded = adapter.extract_refined_gll_leading_edge_frame(
        data, plan=plan, threshold=0.1, extraction_method=method
    )
    restricted = adapter.extract_refined_gll_leading_edge_frame(
        data, plan=plan, threshold=0.1, extraction_method=method, x_min=0.0
    )
    assert np.all(unbounded.success_mask)
    np.testing.assert_array_equal(unbounded.x_front, 0.0)
    assert not np.any(restricted.success_mask)
    np.testing.assert_array_equal(restricted.crossing_count, 0)
    _assert_equal(restricted, _direct(plane, method, threshold=0.1, x_min=0.0))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("option,value", [
    ("extraction_method", "unknown"), ("extraction_method", None),
    ("threshold", np.nan), ("threshold", np.inf), ("threshold", "bad"),
    ("x_min", np.nan), ("x_min", np.inf), ("x_min", True), ("x_min", "bad"),
    ("x_min", 2.0), ("x_min", 1.99),
])
def test_invalid_arguments_match_existing_dispatch_errors(
    context: tuple, monkeypatch: pytest.MonkeyPatch, method: str, option: str, value: object,
) -> None:
    data, plan, plane = context
    monkeypatch.setattr(adapter, "apply_refined_gll_horizontal_slice_plan", lambda *_: plane)
    options = dict(threshold=0.2, extraction_method=method, x_min=None)
    options[option] = value
    direct_options = dict(options)
    direct_method = direct_options.pop("extraction_method")
    with pytest.raises(ValueError) as expected:
        _direct(plane, direct_method, **direct_options)
    with pytest.raises(type(expected.value)) as actual:
        adapter.extract_refined_gll_leading_edge_frame(data, plan=plan, **options)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("method", ("  RIGHTMOST-CROSSING  ", "MOORE-BOUNDARY"))
def test_method_normalization_is_owned_by_existing_dispatch(context: tuple, method: str) -> None:
    data, plan, plane = context
    _assert_equal(
        adapter.extract_refined_gll_leading_edge_frame(
            data, plan=plan, threshold=0.2, extraction_method=method
        ),
        _direct(plane, method, threshold=0.2),
    )


def test_nonperiodic_plane_retains_dispatch_rejection(
    context: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, plan, template = context
    plane = replace(template, periodic_y=False)
    monkeypatch.setattr(adapter, "apply_refined_gll_horizontal_slice_plan", lambda *_: plane)
    with pytest.raises(ValueError, match="periodic_y must be True"):
        adapter.extract_refined_gll_leading_edge_frame(
            data, plan=plan, threshold=0.2, extraction_method="rightmost-crossing"
        )
