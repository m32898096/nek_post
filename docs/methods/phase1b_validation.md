# Phase 1B independent numerical validation specification

This is a proposed specification, not an executed validation report. Phase 1A
only collected evidence and documented the unchanged method. Read the
[method definitions](vortex_numerical_method.md) and
[provenance limits](nek5000_provenance.md) before choosing a reference.

## Reference identity and acceptance policy

First seek the source/build/mesh/executable chain for the production case. If
it cannot be recovered, explicitly name the fingerprinted local `20.0-dev`
source as the reference and keep original-production equivalence unverified.
The historical channel binary and GC_small practice case cannot silently fill
that gap.

Any future harness must run in a new, explicit validation location, preserve
production snapshots/outputs/checkpoints, and record source bytes, build flags,
compiler/library versions, precision, geometry, BCs, physical times, element
maps, field conventions, and reference-generation commands. Phase 1A does not
generate that harness or its fixtures.

Separate two acceptance questions:

1. **Independent method correctness:** compare mathematics and solver/reference
   implementations with tolerances justified by conditioning, storage precision,
   and summation order.
2. **Behavior preservation:** require exact arrays, schemas, filenames, and
   operation order where the existing arithmetic is unchanged. Existing
   regression oracles answer this question without proving solver provenance.

Do not silently replace a golden reference after a mismatch. Preserve the
mismatch and determine whether it comes from geometry, precision, topology,
algorithm, scientific method, or a defect. A scientific change requires its own
review and provenance boundary.

## Comparison sequence

| Stage | Independent comparison | Acceptance evidence |
| --- | --- | --- |
| GLL rule | Export `ZWGLL` nodes/weights and `DGLL` from the pinned compiled source; independently check polynomial moments and summation-by-parts | Node count versus degree explicit; norms and justified errors for multiple degrees |
| Geometry | Export solver coordinates, Jacobians, metrics, and `bm1`; compare against analytic affine/skew/local-curved maps | Separate internal-geometry and float32-coordinate-quantized comparisons; physical axes and orientation explicit |
| Local derivatives | Compare `local_grad3`, `gradm1`, and all nine `comp_gije` entries before recovery | Constant/linear/tensor-polynomial and rotation-plus-strain fields; catch component permutations and curl sign |
| Node equivalence | Export solver global equivalence labels and BC partners; compare contributor sets using physical coordinates and periodic translations | Faces, edges, corners, doubly periodic seams; reversed/permuted/shuffled local layouts; independent boundary/connectivity evidence must establish the caller's connected-interface and translational-periodicity assumptions; unsupported topology is not guaranteed to be detected or rejected |
| Mass assembly | Compare each local mass, assembled denominator, weighted numerator, reciprocal mass, and recovered scalar/tensor | Unequal element volumes; independent contributor-list oracle; conservation, constants, idempotence, and weighted orthogonality |
| Vorticity | Compare raw curl and recovered first component of `comp_vort3` | Signed `dw/dy-dv/dz`; distinguish metric error from assembly error and float64 rounding |
| Projected tensor | Instrument component-wise `bm1 -> dssum -> binvm1` on the same nine gradients | Same topology/masses for every component; this is an explicit validation harness, not a claim that `comp_gije` already projects |
| Eigenvalues | Feed identical fixed tensors to independent eigenvalue evaluation and Python | Zero/shear/real strain, rotation about each axis, trace offsets, near-defective matrices; retain small imaginary parts under the current method |
| Nonlinear order | Construct discontinuous element gradients with known weighted means | Demonstrate `lambda_ci(P(A))` differs from `P(lambda_ci(A))`; lock the intended order |
| Custom Fortran method | Compare its raw invariant calculation and scalar filter separately | Report unprojected versus projected tensors and filtering separately; do not require end-to-end equality with Python |
| Dense P7 | Compare component reconstruction with an independent pointwise basis evaluator | All nine components, permutations/orientations, shared and periodic faces, scalar-curl linearity, then eigenvalues; include a counterexample to scalar lambda_ci interpolation |
| Crop and export | Compare cropped results with the same physical subset of a full-domain calculation | Full-domain recovery retained; coordinates, topology, dtype, time, fields, and filenames preserved |

For different meshes, evaluate at common physical coordinates using appropriate
element-local interpolation/inverse mapping. Equal element counts or filenames
do not define corresponding points or times. Pair stored physical times and
report temporal offsets; any temporal interpolation must be explicit and must
not be mixed into a spatial-operator comparison.

The global mapper checks geometry and structured tiling; rejection tests should
cover violations actually detected by those checks, such as missing/duplicate
cells or inconsistent physical/array-axis mappings. General skew/curved meshes,
nonconforming interfaces, rotational periodicity, and intentionally disconnected
coincident interfaces remain unsupported. The caller must establish connected
interfaces and pure translational periodicity using independent solver boundary
and connectivity evidence before invoking recovery. The mapper does not inspect
solver boundary-condition metadata or independently establish physical
connectivity from external mesh topology. Disconnected coincident interfaces
may therefore be merged, and rotational periodicity is not necessarily detected
or rejected. Test local differentiation on broader valid geometry separately
from these global checks and caller responsibilities. A local curved-element
success is not global mesh support.

## Existing tests to retain

The following are inspected test contracts, not new Phase 1A pass claims:

| Existing file | Relevant contract |
| --- | --- |
| [test_gll.py](../../tests/test_gll.py), [test_gll_differentiation.py](../../tests/test_gll_differentiation.py) | Tabulated GLL values, quadrature exactness, polynomial differentiation, Legendre formula, SBP |
| [test_spectral_derivatives.py](../../tests/test_spectral_derivatives.py) | Physical tensor convention, affine/curved local maps, conditioning, input preservation |
| [test_l2_projection.py](../../tests/test_l2_projection.py) | Independent coordinate/contributor oracle, unequal masses, periodic faces/edges/corners, conservation and idempotence |
| [test_scalar_projection.py](../../tests/test_scalar_projection.py) | Signed scalar recovery and exact comparison with frozen pre-refactor nine-component arithmetic |
| [test_vorticity.py](../../tests/test_vorticity.py) | Curl sign, physical-axis mappings, two-derivative route, independent assembly |
| [test_swirling_strength.py](../../tests/test_swirling_strength.py) | Analytic tensors, projection-before-eigenvalues, small imaginary parts and failure behavior |
| [test_lambda_ci_workflow.py](../../tests/test_lambda_ci_workflow.py), [test_lambda_ci_plotting.py](../../tests/test_lambda_ci_plotting.py) | Workflow order and tensor-before-eigenvalue slice reconstruction |
| [test_vortex_3d.py](../../tests/test_vortex_3d.py), [test_vortex_3d_reconstruction.py](../../tests/test_vortex_3d_reconstruction.py), [test_vortex_3d_regressions.py](../../tests/test_vortex_3d_regressions.py) | Native/combined routes, reconstruction, full-domain periodic contributions before crop, element maps, topology and round-trips |

Candidate additions belong in a separately reviewed reference harness and
small reference fixtures after provenance and acceptance criteria are agreed.
No existing test or golden reference is changed in Phase 1A.

## Independent review criteria

An independent reviewer should verify the reference source identity, actual
compiled precision, coordinate/element-map alignment, every intermediate mass
and derivative comparison, and the distinction between method differences and
defects. A second implementation must not obtain its expected values by calling
the same kernel under test. Report component-wise maxima, relevant weighted
norms, location/element identity of extrema, and failure cases alongside aggregate
errors.

Numerical equivalence, serialized-array fidelity, application compatibility,
and behavior-preserving refactoring are separate conclusions. Reference fields
alone do not validate Tecplot/ParaView behavior, and successful strict resume
does not establish an independent numerical oracle.
