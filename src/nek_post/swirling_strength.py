"""Three-dimensional swirling strength from Cartesian velocity gradients.

``A[..., i, j] = d(velocity_i)/d(physical_coordinate_j)``. The scientific
workflow supplies the *projected* tensor from ``project_velocity_gradient``
(its result.values); this kernel neither differentiates nor projects data.

Following Zhou et al. (1999), JFM 387, Section 2.3, lambda_ci is the largest
absolute imaginary part of the three eigenvalues. All-real spectra give zero.
No imaginary-part tolerance, symmetrization or trace subtraction is applied.
Near repeated/defective eigenvalues, the computed spectrum retains NumPy's
floating-point sensitivity; small nonzero imaginary parts are not filtered.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


DEFAULT_CHUNK_SIZE = 256


@dataclass(frozen=True)
class SwirlingStrengthResult:
    """Float64 scalar fields with the input's leading spatial dimensions.

    Units are inverse time for lambda_ci and inverse time squared for its
    square. A single (3,3) input produces two zero-dimensional NumPy arrays.
    """

    lambda_ci: NDArray[np.float64]
    lambda_ci_squared: NDArray[np.float64]


def compute_swirling_strength(
    velocity_gradient: object, *, chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> SwirlingStrengthResult:
    """Compute lambda_ci and its square from real tensors of shape (...,3,3).

    Native input is (nelem,nt,ns,nr,3,3), producing (nelem,nt,ns,nr) fields.
    Other nonempty spatial shapes and a single matrix are also supported.
    Chunks contain at most chunk_size entries of the first spatial dimension
    (elements for native data). Float32 inputs are converted only per chunk;
    existing float64 inputs are not copied wholesale. Noncontiguous inputs
    are accepted and preserved. np.linalg.eigvals computes only eigenvalues,
    with no full-domain complex result or eigenvector array retained.

    NaN/Inf, complex/non-numeric inputs and malformed/empty shapes raise
    ValueError. Eigensolver failure raises np.linalg.LinAlgError with chunk
    context; non-finite eigenvalues/results or squared overflow raise
    FloatingPointError. Squared underflow follows float64 IEEE arithmetic.
    Callers must supply projected gradients for the scientific workflow;
    anonymous arrays cannot establish their projection provenance.
    """
    if not isinstance(chunk_size, Integral) or isinstance(chunk_size, (bool, np.bool_)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    chunk_size = int(chunk_size)
    try:
        array = np.asarray(velocity_gradient)
    except (ValueError, TypeError) as exc:
        raise ValueError("Velocity gradients must contain real numeric values.") from exc
    if array.ndim < 2 or array.shape[-2:] != (3, 3) or any(n == 0 for n in array.shape[:-2]):
        raise ValueError("Velocity gradients must have nonempty shape (..., 3, 3).")
    if array.dtype.kind not in "fiu":
        raise ValueError("Velocity gradients must contain real numeric values.")
    spatial_shape = array.shape[:-2]
    if not spatial_shape:
        array = array[None]
    count = array.shape[0]
    lambda_ci = np.empty(array.shape[:-2], dtype=np.float64)
    squared = np.empty_like(lambda_ci)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        with np.errstate(over="ignore", invalid="ignore"):
            chunk = np.asarray(array[start:stop], dtype=np.float64)
        if not np.all(np.isfinite(chunk)):
            raise ValueError(f"Velocity gradients must be finite in chunk [{start}:{stop}].")
        try:
            eigenvalues = np.linalg.eigvals(chunk)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(f"Eigenvalue calculation failed in chunk [{start}:{stop}].") from exc
        if not np.all(np.isfinite(eigenvalues)):
            raise FloatingPointError(f"Non-finite eigenvalues in chunk [{start}:{stop}].")
        strength = np.max(np.abs(eigenvalues.imag), axis=-1)
        lambda_ci[start:stop] = strength
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            np.square(strength, out=squared[start:stop])
        if not np.all(np.isfinite(squared[start:stop])):
            raise FloatingPointError(f"lambda_ci_squared exceeds float64 range in chunk [{start}:{stop}].")
        # Release complex and converted buffers before the next eigensolve.
        del chunk, eigenvalues, strength
    return SwirlingStrengthResult(lambda_ci.reshape(spatial_shape), squared.reshape(spatial_shape))
