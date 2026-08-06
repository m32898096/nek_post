from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal
import pytest

import nek_post.leading_edge_methods.moore_boundary as moore
from nek_post.leading_edge_extraction import extract_spanwise_leading_edge
from nek_post.leading_edge_methods import (
    DEFAULT_LEADING_EDGE_METHOD,
    SUPPORTED_LEADING_EDGE_METHODS,
    extract_leading_edge,
    normalize_leading_edge_method,
    restrict_leading_edge_x_domain,
)


def test_supported_method_names_are_immutable_and_canonical() -> None:
    assert SUPPORTED_LEADING_EDGE_METHODS == (
        "rightmost-crossing",
        "moore-boundary",
    )
    assert DEFAULT_LEADING_EDGE_METHOD == "rightmost-crossing"
    assert normalize_leading_edge_method("rightmost-crossing") == (
        "rightmost-crossing"
    )
    assert normalize_leading_edge_method("  RIGHTMOST-CROSSING  ") == (
        "rightmost-crossing"
    )
    assert normalize_leading_edge_method("MOORE-BOUNDARY") == "moore-boundary"


@pytest.mark.parametrize(
    "value",
    ["unknown", "", True, False, 1, None, ["rightmost-crossing"]],
)
def test_method_validation_rejects_unsupported_or_malformed_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="method"):
        normalize_leading_edge_method(value)


def test_dispatcher_is_exactly_equivalent_to_legacy_extractor() -> None:
    x = np.arange(7, dtype=np.float64)
    y = np.array([0.0, 0.25, 0.5, 0.75])
    concentration = np.array(
        [
            [-1.0, 0.3, -0.2, 0.5, -0.4, 0.3, -1.0],
            [-1.0, 0.1, 0.1, 0.1, -1.0, -1.0, -1.0],
            [np.nan, -1.0, 0.4, np.nan, -1.0, 0.4, -1.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    legacy = extract_spanwise_leading_edge(
        x, y, concentration, threshold=0.1
    )
    dispatched = extract_leading_edge(
        x,
        y,
        concentration,
        threshold=0.1,
        method="rightmost-crossing",
        periodic_y=True,
        y_period=1.0,
    )

    assert dispatched.method == "rightmost-crossing"
    assert dispatched.threshold == legacy.threshold
    assert_array_equal(dispatched.y, legacy.y)
    assert_array_equal(dispatched.success_mask, legacy.success_mask)
    assert_array_equal(dispatched.crossing_count, legacy.crossing_count)
    assert_array_equal(np.isnan(dispatched.x_front), np.isnan(legacy.x_front))
    finite = np.isfinite(legacy.x_front)
    assert_array_equal(dispatched.x_front[finite], legacy.x_front[finite])
    for values in (
        dispatched.y,
        dispatched.x_front,
        dispatched.success_mask,
        dispatched.crossing_count,
    ):
        assert not values.flags.writeable


@pytest.mark.parametrize(
    "period", [0.0, -1.0, np.inf, -np.inf, np.nan, None, True]
)
def test_periodic_context_rejects_invalid_period(period: object) -> None:
    with pytest.raises(ValueError, match="y_period"):
        extract_leading_edge(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.5]),
            np.zeros((2, 2)),
            threshold=0.1,
            periodic_y=True,
            y_period=period,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("periodic_y", [False, 0, 1, "True", None])
def test_periodic_context_requires_explicit_true_boolean(
    periodic_y: object,
) -> None:
    with pytest.raises(ValueError, match="periodic_y"):
        extract_leading_edge(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.5]),
            np.zeros((2, 2)),
            threshold=0.1,
            periodic_y=periodic_y,  # type: ignore[arg-type]
            y_period=1.0,
        )


def test_periodic_endpoint_must_remain_excluded() -> None:
    with pytest.raises(ValueError, match="exclude"):
        extract_leading_edge(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.5, 1.0]),
            np.zeros((3, 2)),
            threshold=0.1,
            periodic_y=True,
            y_period=1.0,
        )


def test_dispatcher_selects_moore_boundary_grid_node_result() -> None:
    x = np.arange(5, dtype=float)
    y = np.arange(4, dtype=float) / 4.0
    concentration = np.broadcast_to(
        np.where(x[None, :] < 2.0, 1.0, 0.0),
        (y.size, x.size),
    ).copy()

    result = extract_leading_edge(
        x,
        y,
        concentration,
        threshold=0.5,
        method="moore-boundary",
        periodic_y=True,
        y_period=1.0,
    )

    assert result.method == "moore-boundary"
    assert_array_equal(result.x_front, np.full(y.size, 2.0))


def test_strict_x_domain_removes_negative_and_exact_bound_columns() -> None:
    x = np.array([-1.0, 0.0, 0.25, 1.0])
    concentration = np.arange(8, dtype=float).reshape(2, 4)

    x_work, concentration_work = restrict_leading_edge_x_domain(
        x,
        concentration,
        x_min=0.0,
    )

    assert_array_equal(x_work, [0.25, 1.0])
    assert_array_equal(concentration_work, concentration[:, 2:])
    assert not x_work.flags.writeable
    assert not concentration_work.flags.writeable


def test_moore_domain_restriction_precedes_candidate_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[np.ndarray] = []

    def candidates(values: object, *, threshold: float) -> np.ndarray:
        seen.append(np.asarray(values).copy())
        return np.zeros(np.asarray(values).shape, dtype=bool)

    monkeypatch.setattr(moore, "build_low_side_boundary_candidate_mask", candidates)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.arange(4, dtype=float) / 4.0
    field = np.broadcast_to(np.arange(x.size), (y.size, x.size)).copy()

    extract_leading_edge(
        x,
        y,
        field,
        threshold=0.5,
        method="moore-boundary",
        periodic_y=True,
        y_period=1.0,
        x_min=0.0,
    )

    assert len(seen) == 1
    assert_array_equal(seen[0], field[:, 3:])


@pytest.mark.parametrize(
    ("method", "expected_x"),
    [("rightmost-crossing", 2.5), ("moore-boundary", 3.0)],
)
def test_symmetric_periodic_front_uses_only_positive_x_domain(
    method: str, expected_x: float
) -> None:
    x = np.arange(-4.0, 5.0)
    y = np.arange(6, dtype=float) / 6.0
    field = np.broadcast_to((np.abs(x) < 3.0)[None, :], (y.size, x.size)).astype(float)

    result = extract_leading_edge(
        x,
        y,
        field,
        threshold=0.5,
        method=method,
        periodic_y=True,
        y_period=1.0,
        x_min=0.0,
    )

    assert_array_equal(result.x_front, np.full(y.size, expected_x))
    assert np.all(result.x_front[result.success_mask] > 0.0)


def test_rightmost_restriction_preserves_crossing_algorithm() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 0.5])
    field = np.array(
        [[0.0, 1.0, 0.0, 0.8, 0.2, 0.0], [1.0, 0.0, 1.0, 0.9, 0.1, 0.0]]
    )
    expected = extract_spanwise_leading_edge(
        x[x > 0.0], y, field[:, x > 0.0], threshold=0.5
    )

    result = extract_leading_edge(
        x,
        y,
        field,
        threshold=0.5,
        method="rightmost-crossing",
        periodic_y=True,
        y_period=1.0,
        x_min=0.0,
    )

    assert_array_equal(result.x_front, expected.x_front)
    assert_array_equal(result.crossing_count, expected.crossing_count)


@pytest.mark.parametrize(
    ("x", "message"),
    [([-1.0, 0.0], "no retained"), ([-1.0, 0.0, 1.0], "at least two")],
)
def test_insufficient_strict_x_domain_is_rejected(
    x: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        restrict_leading_edge_x_domain(
            x,
            np.zeros((2, len(x))),
            x_min=0.0,
        )
