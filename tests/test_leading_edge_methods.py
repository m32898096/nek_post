from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from nek_post.leading_edge_extraction import extract_spanwise_leading_edge
from nek_post.leading_edge_methods import (
    DEFAULT_LEADING_EDGE_METHOD,
    SUPPORTED_LEADING_EDGE_METHODS,
    extract_leading_edge,
    normalize_leading_edge_method,
)


def test_supported_method_names_are_immutable_and_canonical() -> None:
    assert SUPPORTED_LEADING_EDGE_METHODS == ("rightmost-crossing",)
    assert DEFAULT_LEADING_EDGE_METHOD == "rightmost-crossing"
    assert normalize_leading_edge_method("rightmost-crossing") == (
        "rightmost-crossing"
    )
    assert normalize_leading_edge_method("  RIGHTMOST-CROSSING  ") == (
        "rightmost-crossing"
    )


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
