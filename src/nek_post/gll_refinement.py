"""Element-local post-processing resampling of existing GLL polynomials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nek_post.fields import get_concentration, get_coordinates
from nek_post.gll import gll_interpolation_matrix


@dataclass(frozen=True)
class RefinedGLLElement:
    """One source element evaluated on a target GLL nodal set.

    Arrays returned by the adapters are immutable float64 tensors in the
    original NumPy axis order. ``target_nodal_order = target_node_count - 1``
    describes only the target nodal set, not a new simulation polynomial order.
    The source polynomial and its solution information are unchanged.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    concentration: NDArray[np.float64]
    source_node_count: int
    target_node_count: int
    source_polynomial_order: int
    target_nodal_order: int


def refine_gll_tensor3(
    values: object,
    *,
    target_node_count: int,
) -> NDArray[np.float64]:
    """Evaluate one element's tensor polynomial on a target GLL nodal set.

    ``values`` must be a finite real 3-D tensor with at least two GLL nodes
    per axis, ordered as ascending reference nodes in each NumPy axis. Source
    node counts are inferred independently from its shape; axis order is
    preserved without assigning physical x/y/z directions. The returned float64
    array has shape ``(target_node_count,) * 3`` and does not modify the input.

    ``node_count`` means GLL node count; ``polynomial_order = node_count - 1``.
    In particular, (8, 8, 8) -> (10, 10, 10) evaluates the same P7 spectral-element
    polynomial on the order-9 GLL nodal set. This is post-processing resampling,
    not a true N9 simulation or new solution information. A smaller target count
    is allowed, but its nodal values need not retain the full source polynomial.
    """
    if np.iscomplexobj(values):
        raise ValueError("values must contain only real values.")
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 3:
        raise ValueError("values must be a three-dimensional tensor.")
    if any(count < 2 for count in values_arr.shape):
        raise ValueError("values must have at least 2 GLL nodes per axis.")
    if not np.all(np.isfinite(values_arr)):
        raise ValueError("values must contain only finite values.")

    matrix0, matrix1, matrix2 = (
        gll_interpolation_matrix(source_node_count, target_node_count)
        for source_node_count in values_arr.shape
    )
    return np.einsum(
        "ai,bj,ck,ijk->abc",
        matrix0,
        matrix1,
        matrix2,
        values_arr,
        optimize=True,
    )


def _element_array(values: object, name: str) -> NDArray[np.float64]:
    try:
        if np.iscomplexobj(values):
            raise ValueError("Complex values are not supported.")
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Element {name} must be a real numeric array.") from exc
    if array.ndim != 3:
        raise ValueError(
            f"Element {name} must be three-dimensional; got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Element {name} must contain only finite values.")
    return array


def _immutable_tensor(values: NDArray[np.float64]) -> NDArray[np.float64]:
    # Immutable bytes backing also prevents callers from re-enabling writes.
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)


def refine_nek_element_gll(
    element: object,
    *,
    target_node_count: int,
) -> RefinedGLLElement:
    """Resample coordinates and concentration of one isotropic Nek element.

    All four source arrays must have the same finite, real, three-dimensional
    shape with at least two GLL nodes per axis. Coordinates and concentration
    are obtained through the standard field accessors. Each tensor is evaluated
    independently using :func:`refine_gll_tensor3`, preserving source axis order.

    For N7, 8 source nodes per axis and 10 target nodes produce four (10, 10, 10)
    arrays representing the same P7 geometry/field polynomials. This is solely
    post-processing, not a true N9 solution. Smaller target counts are allowed
    but need not retain the full source polynomial. Returned arrays are float64,
    deterministic, read-only, and independent of the source arrays.
    """
    try:
        coordinates = get_coordinates(element)
    except Exception as exc:
        raise ValueError(f"Element does not provide complete coordinates: {exc}") from exc
    try:
        concentration = get_concentration(element)
    except Exception as exc:
        raise ValueError(f"Element does not provide concentration: {exc}") from exc

    names = ("x", "y", "z", "concentration")
    arrays = tuple(
        _element_array(values, name)
        for name, values in zip(names, (*coordinates, concentration), strict=True)
    )
    source_shape = arrays[0].shape
    for name, array in zip(names[1:], arrays[1:], strict=True):
        if array.shape != source_shape:
            raise ValueError(
                f"Element {name} shape must match x shape {source_shape}; "
                f"got {array.shape}."
            )
    if any(count < 2 for count in source_shape):
        raise ValueError("Element must have at least 2 GLL nodes per axis.")
    if len(set(source_shape)) != 1:
        raise ValueError(
            f"Element source shape must be isotropic; got {source_shape}."
        )

    x, y, z, refined_concentration = (
        _immutable_tensor(refine_gll_tensor3(array, target_node_count=target_node_count))
        for array in arrays
    )
    source_node_count = source_shape[0]
    return RefinedGLLElement(
        x=x,
        y=y,
        z=z,
        concentration=refined_concentration,
        source_node_count=source_node_count,
        target_node_count=int(target_node_count),
        source_polynomial_order=source_node_count - 1,
        target_nodal_order=int(target_node_count) - 1,
    )


def refine_nek_elements_gll(
    data: object,
    *,
    target_node_count: int,
) -> tuple[RefinedGLLElement, ...]:
    """Resample a nonempty ``data.elem`` collection in its original order.

    Each element must be isotropic, but may have its own source node count.
    Elements remain separate: no global grid or interface deduplication is
    performed. Malformed elements are reported with their zero-based index.
    """
    try:
        elements = tuple(data.elem)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise ValueError("Nek data must provide an iterable 'elem' collection.") from exc
    if not elements:
        raise ValueError("Nek data contains no spectral elements.")
    refined = []
    for index, element in enumerate(elements):
        try:
            refined.append(
                refine_nek_element_gll(element, target_node_count=target_node_count)
            )
        except ValueError as exc:
            raise ValueError(f"Element {index}: {exc}") from exc
    return tuple(refined)
