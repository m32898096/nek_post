# Single-frame 3D binary VTU: native grid and P7 reconstruction

`scripts/40_export_vortex_3d.py` exports exactly one original snapshot as one
`vtkUnstructuredGrid`, with float64 PointData `omega_x` and `lambda_ci` on
the same physical points and cells. The default remains the Phase 1 native-grid
representation. Phase 2 adds optional element-local P7 reconstruction on denser
GLL nodal sets. There is no batch, PVD, resume, overwrite or smoothing option.
The Phase 1 description below still applies to the default path; the
[Phase 2 section](#phase-2-p7-volume-reconstruction) specifies denser sampling.

## Numerical sequence and assumptions

The exporter reuses `element_velocity_gradient` from `spectral_derivatives.py`:
the existing float64 GLL differentiation matrices, native `(nt,ns,nr)` layout
with `(r,s,t)` on array axes `(2,1,0)`, geometry Jacobian and physical gradient
solve. Tensor rows are `(u,v,w)`, columns physical `(x,y,z)`:
`A[...,i,j] = du_i/dx_j`. Each element's full tensor is computed once.

The existing pipelines have different operation orders, preserved here:

| Field | Sequence from the common local tensor A |
| --- | --- |
| Signed streamwise vorticity | `omega_x = P(A[...,2,1] - A[...,1,2])` |
| Swirling strength | `lambda_ci = max(abs(imag(eig(P(A)))))` |

`P` is the validated mass-lumped GLL L2 projection using original nodal
Jacobians, quadrature weights and the explicit periodic node map. The scalar
projection and nine-component tensor projection functions are reused directly.
This recovery is the established scientific method, including its mass-weighted
shared-node assembly; the visualization adds no averaging. Eigenvalues use the
existing `compute_swirling_strength` without filtering small imaginary parts,
trace subtraction or symmetrization. Both fields have inverse-time units.

The dedicated omega-only implementation solves for two inverse-Jacobian columns
and forms the curl locally before scalar projection. This exporter extracts
those derivatives from the full tensor solve before the same scalar projection.
The mathematical definition and projection order agree, but floating-point
operation order differs: bitwise equality with the optimized omega-only path
is not promised. Synthetic regression tests compare them with a tight numeric
tolerance, an independent coordinate-based mass oracle, and exact lambda_ci
equality against the existing tensor pipeline. No direct Nek5000-generated
vorticity reference or real-frame numerical comparison was run for this feature.

The supported global geometry remains a complete affine axis-aligned structured
box, with one common local node shape and physical/array-axis mapping, possibly
permuted axes or reversed element orientations. The existing mapper checks GLL
profiles with its storage-aware tolerance. An additional existing exact geometry
check rejects mismatched coordinate replicas and neighboring endpoints. Original
stored coordinates are retained, promoted to float64 without adjustment. Curved,
skew, nonconforming, disconnected coincident and rotational-periodic meshes
are outside this prototype's global projection scope. Translational periodicity
must be explicitly confirmed by the caller; it cannot be inferred from geometry.
Defaults `(x,y)` and `(272,12,8)` physical element counts are for `N7_H` only.

### Element maps and canonical configuration

The installed Pymech 2.0.1 `read_header`, `readnek`, `writenek` and `HexaData`
implement **one-based global element IDs**. An unsplit snapshot stores its
signed int32 element map after the 132-byte header and four-byte float32 endian
tag. File records follow that map, while Pymech puts record `ID` into
`data.elem[ID-1]`. Consequently, exported element order is ascending reader
global ID, not necessarily raw file-record order. The exporter does not apply
the file permutation a second time.

Preflight now reads and validates that original map before payload loading.
For the supported complete unsplit snapshot it must be exactly a permutation
of `1..N`. Missing/truncated, duplicate, zero, negative and out-of-range IDs
are rejected, including maps that would trigger Pymech's sequential fallback.
The map is checked again before payload loading and compared against the map
retained by the reader before any scientific calculation. Provenance records
the convention and SHA-256 of the file-order IDs serialized canonically as
little-endian int32; the complete source hash still covers original file bytes.
Both byte orders and valid nontrivial permutations have synthetic tests.

At each public workflow entry, `periodic_axes`, `expected_element_counts` and
crop bounds are normalized into one frozen configuration with immutable
tuples. Inspection, calculation and provenance use that same configuration.
User generators/iterators are consumed once; lists, tuples, generators and
iterators produce identical numerical arrays and configuration metadata.

## Mesh and crop semantics

For each SEM element, every native GLL point is retained. The exporter creates
`(nt-1)*(ns-1)*(nr-1)` linear `VTK_HEXAHEDRON` subcells; a P7/GLL8 element has
512 points and **343 cells**, including its interior. It never substitutes one
eight-corner hex for a full spectral element. Connectivity uses VTK's positive
physical x/y/z winding, determined from validated geometry rather than an
assumed correspondence between physical and reference axes.

Points remain element-major, with `(t,s,r)` C ordering and r fastest within
each element. Shared-face/edge/corner point IDs are deliberately **not merged**,
even when projected field values agree. Both physical endpoints of periodic
directions remain distinct. There is no visualization-level coordinate or
field averaging. Consequences: coincident interface points consume more space,
and filters that depend on connected topology (surface extraction, connectivity,
streamlines or new derivatives) may treat SEM elements as separate regions or
expose internal faces. Do not use a viewer's Clean/Merge filter as validation
without a separate coordinate/field agreement audit. Native subcells within
each SEM element share points normally.

`--crop XMIN XMAX YMIN YMAX ZMIN ZMAX` keeps only subcells whose eight native
vertices are inside the inclusive physical box. The crop must be finite,
strictly ordered, within the domain, and contain at least one full 3D subcell.
Boundary locations generally recede to existing GLL coordinates. No points are
created at requested crop edges; the actual exported bounds are recorded.
Used points retain their original relative order. Derivatives and L2 recovery
always use the **entire original domain before cropping**, preserving shared
and periodic contributions. A crop reduces output size, not full-domain
gradient memory or input I/O.

These VTU cells interpolate fields linearly between nonuniform GLL samples.
They do not encode the native P7 basis, preserve all inter-node extrema, or
establish converged isosurfaces, contours or spatial accuracy.

### Hexahedral validity before serialization

`hex_validation.py` checks cell types, offsets, exactly eight in-range distinct
point IDs per cell, distinct physical vertices, and positive nondegenerate
orientation before a VTU file is created. It works in chunks of 4096 cells,
using native VTK connectivity/offset arrays. Exact x/y/z cuboids are verified
analytically: all eight positions must match VTK corner order and the three
finite edge lengths must be positive. This accepts nonuniform, strongly
anisotropic GLL subcells without an absolute-volume cutoff.

The standalone writer also supports skew affine and warped **trilinear**
hexes. Each Jacobian column is evaluated as a bilinear combination of its four
reference-direction edges, normalized separately by that direction's maximum
absolute edge component. This avoids translation cancellation and an isotropic
volume threshold. For a trilinear hex, `det(J)` is a tensor polynomial of degree
at most `(2,2,2)`. Its 27 values on `{0,1/2,1}^3` determine the complete
polynomial, which is converted to tensor Bernstein coefficients. Positive
coefficients bound the determinant away from zero over the entire cube because
the Bernstein basis is nonnegative and sums to one. Inconclusive bounds undergo
de Casteljau subdivision; positivity at the 27 samples alone is not accepted.
Tests include an interior fold missed by all 27 positive samples and positive
integrated volume, plus a valid twisted cell requiring subdivision to certify.

The normalized determinant must exceed `1e-12`; this float64 margin is not
formal interval arithmetic. Subdivision is limited to eight levels and 4096
visited boxes per cell. Numerically degenerate or uncertifiable geometry fails
closed with an explicit error. A valid cell extremely close to singularity can
therefore be refused. The check certifies local orientation of the trilinear
subcell map, not global non-overlap between cells or the original curved P7
mapping between native nodes. It moves no points and evaluates no scalars.
Accepting warped standalone VTU cells does **not** extend the full workflow's
existing affine structured-mesh projection scope to general curved SEM meshes.

## Memory and performance limits

No projection arithmetic or scientific kernel was optimized or changed.
Computation still retains complete local raw and projected nine-component
tensors during projection. The existing swirling-strength kernel also computes
its squared scalar, which this export does not retain; no alternate eigensolver,
filter or scalar definition has been introduced.

For the documented N7_H dimensions (`26112` elements, `8^3` nodes each), the
following are array-size estimates, **not measured real-data peak RSS**:

| Allocation | Approximate GiB |
| --- | ---: |
| One float64 nine-component gradient tensor | 0.896 |
| Raw and projected tensors together | 1.793 |
| One local float64 scalar or local GLL mass weights | 0.100 |
| Original element-local float64 XYZ export points | 0.299 |
| Both exported float64 PointData arrays | 0.199 |
| One eight-int64-ID connectivity buffer, `8956416` cells | 0.534 |
| One packed nine-int64-per-cell representation | 0.601 |

The current mesh builder temporarily retains per-element lists, concatenated
connectivity and packed connectivity, and VTK may allocate its own topology.
These are additional costs, not alternative representations to count only once.
The writer's exact read-back comparison now uses native offsets and connectivity
directly, avoiding two extra full packed-cell arrays (about 1.20 GiB together
for this case), with regression coverage for altered IDs and altered offsets.
File decoding, compression buffers, coordinates, scalars, projection masses,
reader storage and validation scratch add further memory. Jacobian validation
scratch is bounded by its chunk size and does not allocate a full-domain
Jacobian or determinant tensor.

Preflight retains its conservative formula, where `N` is local points, `C` is
subcells, `V` is the sum of header variable counts and `S` is source bytes:

```text
B = 40*N + 80*C + 1 MiB
RAM = 8*V*N + S + 240*N + 6*B + 1 GiB
disk = 2*B + 1 MiB
```

With eight stored variables and float32 input this is roughly **12.2 GiB RAM**
for the documented case, including an estimated source size. This is an
allocation allowance, not a benchmark or memory reservation. Late resource
checks remain conservative while the export grid is retained. Even a small
crop loads the full snapshot, builds full-domain projection tensors, and
computes full-domain eigenvalues before selecting subcells. Invalid crop
coverage can also fail only after that work. Input/output hashing, millions of
cells, binary compression/read-back and pathological warped-cell subdivision
can be expensive. Actual wall time, peak RSS, disk size and viewer memory remain
unmeasured; no real-data benchmark has been run.

## Commands and outputs

Header/map-only preflight (no velocity read, full-source hash pass, VTK reader or output writes):

```bash
PYENV_VERSION=research312 python -B scripts/40_export_vortex_3d.py \
  --case N7_H --index 79 \
  --output-dir /data/Nek5000_data/results/vortex_3d_phase1_review \
  --crop 14 16 0 1.5 0 1 --dry-run
```

The precise proposed real-data validation command, **not run as part of
implementation** and requiring separate authorization for its full-domain
calculation, is:

```bash
PYENV_VERSION=research312 python -B scripts/40_export_vortex_3d.py \
  --case N7_H --index 79 \
  --output-dir /data/Nek5000_data/results/vortex_3d_phase1_review \
  --periodic-axes x y --expected-element-counts 272 12 8 \
  --crop 14 16 0 1.5 0 1
```

Omit `--crop` to export the full domain. `--case` accepts `N7_H`, `case_N7_H`
or an external case-directory path. Default selection is only frame 79.
Use an empty `--periodic-axes` for a confirmed nonperiodic mesh and override
`--expected-element-counts` for other complete physical partitions.
Header/map preflight cannot validate actual geometry, axis orientation or crop
coverage; those checks run after the explicitly requested payload read.

The example creates just:

```text
/data/Nek5000_data/results/vortex_3d_phase1_review/case_N7_H/f00079/
  GC0_f00079.vtu
  GC0_f00079.json
```

Default output root is `/data/Nek5000_data/results/vortex_3d_native`.
The basename follows `/home/dai_lab/convert_nek_to_vtu.py`, inspected as the
file-format reference. That legacy script uses `Nek5000Reader` and
`mesh.save(..., binary=True)`; this exporter constructs the native subcell mesh
itself and uses the same PyVista binary save API. The legacy reader's spatial
representation is not evidence of spectral fidelity. The legacy script is
unchanged.

VTU FieldData `TIME_VALUE` preserves the actual snapshot time, not the index
or an assumed increment. The JSON records source path/SHA-256, frame, time,
step, input precision, geometry/order, projection/crop configuration, scalar
ranges, output SHA-256, implementation file hashes, library versions, resource
estimates, process peak RSS and exact binary round-trip verification. Viewer
verification flags remain false. No PVD or time series is created.

Existing frame directories, symlink output ancestry, repository/raw-data paths,
and paths nested inside scientific frame directories are rejected. The workflow
requires original data and generated outputs outside the repository. Full-domain
memory and conservative disk checks precede computation and writing; estimates
do not reserve resources. All output goes first into a private sibling staging
directory, is read back and compared exactly, and is published together with
the JSON using Linux atomic no-replace directory publication. Source hashes,
stats and implementation signatures are rechecked before publication. Failed
staging is preserved for review. There is no automated repair or cleanup, and
this is not a power-loss durability guarantee or filesystem snapshot.

## Validation and remaining work

Synthetic tests cover signed linear-field curl, known real/complex spectra,
projection before eigenvalues, nonuniform mass weighting and periodic seams,
single evaluation of each native gradient, all six axis permutations/reversed
orientations, interior-node participation, positive hex winding, coordinate-
encoded point ordering, crop semantics, binary XML encoding and exact VTK
round-trip, source/output collisions, changed source, invalid geometry/order,
nonfinite data, low resources and publication failures. Small generated Nek
inputs and VTUs live only in pytest temporary directories outside the repository.
Review regressions additionally cover original/retained element maps, Pymech
fallback rejection, big/little endian and float32/float64 files, all combinations
of iterable configuration types, malformed/degenerate/inverted/warped hexes,
interior folds missed by coarse sampling, full eight-vertex P7 adjacency checks,
and exact cropped/full field equality with periodic boundary contributions.

```bash
PYENV_VERSION=research312 python -m pytest -q tests/test_vortex_3d.py \
  tests/test_vortex_3d_regressions.py \
  tests/test_vorticity.py tests/test_scalar_projection.py tests/test_l2_projection.py \
  tests/test_spectral_derivatives.py tests/test_swirling_strength.py -ra
PYENV_VERSION=research312 python -m compileall -q src scripts tests
PYENV_VERSION=research312 python -m pytest -q -ra
git diff --check
```

Initial implementation verification on 2026-11-07 in `research312`: **285 focused tests
passed**, including **69 new exporter cases**; **1,865 full-suite tests passed
in 50.26 s**, with **zero failures and zero skips**. Compileall, CLI help,
tracked `git diff --check` and equivalent whitespace checks on all five new
files passed. Existing numerical modules and the legacy converter were not
modified. No real snapshot calculation, real-data dry-run, or viewer session
was performed for this feature.

After the review corrections on 2026-11-07: **396 focused tests passed**
(including **180 exporter/mesh/review regression cases**); the final complete
suite passed **1,976 tests in 54.07 s**, with **zero failures and zero skips**.
The initial 29-case review regression set reproduced 23 failures before the
fixes; it was then expanded with map permutations, iterator/mutation cases,
warped-cell certificates, exact P7 adjacency/crops and binary topology checks.
Compileall (bytecode redirected to `/tmp`), CLI help, tracked whitespace checks
and all seven untracked-file whitespace checks passed. The existing nine
numerical/geometry/I-O modules matched HEAD byte for byte. Self-review confirmed
that raw/retained map validation precedes scientific computation, one immutable
configuration drives computation and metadata, cell validation precedes VTU
creation, and cropped scalars exactly retain full-domain projected values.
The final valid-map fixture distinguishes both signed omega_x and lambda_ci
between elements. No real input or benchmark was loaded, and all changes
remain uncommitted for independent review.

VTK reader round-trip is a file/data check. **ParaView and Tecplot import and
rendering have not been verified for this exporter.** After an authorized
real-frame run, open the same VTU in both applications, confirm 3D cell/point
counts, actual bounds, the two PointData arrays, double precision, timestamp,
signed omega_x and lambda_ci ranges, and inspect slices/isosurfaces. Compare
against existing native outputs by physical coordinates, accounting for
duplicate element-local points and the omega solve's floating-point limits.
Keep the original exported mesh and avoid smoothing/merging during validation.
Application behavior, real-data resource estimates and high-order visualization
convergence remain untested here; no compatibility or convergence claim is made.

## Phase 2: P7 volume reconstruction

`--sampling-nodes N` selects an isotropic **target GLL nodal set** for a source
with `(8,8,8)` native nodes. Supported examples are 8, 10 and 12; integer counts
larger than 12 are configurable, subject to preflight resources and int64
topology capacity. Counts below 8, booleans, nonintegers and explicit sampling
on a non-P7 source fail. Omitting the option preserves the native default,
including Phase 1's support for other native node shapes. Explicit 8 uses that
same native path, coordinates, scalar arrays, connectivity and basename.

For a target count `N`, the reference coordinates are `-1`, `+1` and the roots
of the derivative of Legendre `P_(N-1)`. The existing cached
`gll_interpolation_matrix(8,N)` and `refine_gll_tensor3` evaluate the source P7
tensor-product Lagrange polynomial. Ten nodes use the degree-9 GLL *nodal set*,
and twelve the degree-11 set; the **simulation and reconstructed polynomial
remain P7 in each reference direction**. This adds samples, not CFD resolution
or a higher simulation polynomial degree. These target nodes are nonuniform
GLL points, unlike the uniform reference-edge samples in the existing 29-point
slice workflows. The 8/10/12 sets are generally not nested: denser sampling
evaluates new locations rather than simply appending points to the native set.
They are not obtained by subdividing native linear VTU cells.

The common native preparation was factored out of `compute_native_vortex_fields`
without changing its gradient, scalar-projection, tensor-projection or native
eigenvalue arithmetic. Denser output follows:

```text
original full-domain velocity -> nine native Cartesian gradients A
  omega_x: P(A[2,1]-A[1,2]) -> P7 interpolation I -> output omega_x
  lambda_ci: P(A) -> P7 interpolation of all nine components I
                 -> existing eigenvalue kernel -> output lambda_ci
```

Interpolating the already-computed native `lambda_ci` is deliberately excluded:
eigenvalues are nonlinear and that route generally gives different values.
For each reconstructed element, the primary projected-curl scalar is compared
with `I(P(A))[2,1]-I(P(A))[1,2]`. Their maximum absolute difference must be at
most `2e-12 * max(1, max(abs(I(P(A)))), max(abs(I(P(omega_x)))))`.
This matches the existing slice shared-trace tolerance and allows the scalar
and tensor projection summation orders to differ in float64. The alternate
route is an audit only; it never replaces, averages or clips the primary field.

Interpolation is performed in increasing physical `(z,y,x)` order, then
returned to the original element-local array ordering. This fixes contraction
order across reversed elements. Physical coordinate profiles use the same P7
interpolation matrix on the **original stored profiles**, including their
storage roundoff; exact rectilinearity permits a factorized one-dimensional
geometry evaluation. Replicas and native endpoints must already agree exactly.
Target profiles must remain finite and strictly increasing. This retains the
existing affine axis-aligned structured geometry scope, not arbitrary curved
or skew SEM geometry. The standalone writer's broader trilinear-cell validator
does not expand that scope.

Every target point participates in positive VTK-ordered linear hexahedral
subcells: `N^3` points and `(N-1)^3` hexes per full SEM element. Points remain
unmerged and in the original element-major C order. Full-domain native
projected tensor/scalar face traces are audited before cropping. Exported
coordinate, `omega_x` and `lambda_ci` faces are compared against their neighbors,
including available translational-periodic partners, with the existing scaled
`2e-12` trace tolerance. Periodic normal coordinates remain the two distinct
original domain endpoints; tangential coordinates and fields must agree.
No shared data are averaged. Trace differences and omega route differences
are recorded in provenance. Near defective spectra retain the existing
eigensolver's roundoff sensitivity; no imaginary-part filtering is introduced.

Cropping retains wholly contained **target** subcells. Geometry planning counts
these cells before dense buffers are allocated. No crop-edge nodes are created;
used points retain their relative original order. Full-domain differentiation
and projection still precede cropping; only dense sampling/eigenvalues for
excluded SEM elements are avoided. Cropped retained fields are exactly the
corresponding full-output samples. The existing writer validates every emitted
hex with unchanged cuboid/Jacobian/Bernstein checks and verifies exact binary
float64 arrays, coordinates, connectivity and `TIME_VALUE` on read-back.

### Selection, output protection and provenance

Example for a future separately authorized single-frame run (not executed):

```bash
PYENV_VERSION=research312 python -B scripts/40_export_vortex_3d.py \
  --case N7_H --index 79 --sampling-nodes 10 \
  --output-dir /data/Nek5000_data/results/vortex_3d_phase2_review \
  --periodic-axes x y --expected-element-counts 272 12 8 \
  --crop 14 16 0 1.5 0 1 --dry-run
```

Denser output adds a sampling directory and filename suffix:

```text
ROOT/gll10/case_N7_H/f00079/GC0_f00079_gll10.{vtu,json}
ROOT/gll12/case_N7_H/f00079/GC0_f00079_gll12.{vtu,json}
```

The default dense root is `/data/Nek5000_data/results/vortex_3d_reconstructed`;
the native default root and names are unchanged. Native, 10 and 12 outputs can
coexist under an explicitly selected root. All existing path validation,
no-replace publication, staged round-trip, source integrity and collision
protection are reused. JSON adds source/target node counts, the reference-node
definition, interpolation convention, field operation order, full estimated
counts, actual crop counts, bounded-workspace estimate and interface diagnostics.
`--dry-run` reads the header/map and resources only: it does not compute target
nodes, decode the payload, verify geometry/crop coverage, or create outputs.

### Resources and deferred convergence validation

One native full-domain raw/projected tensor pair is still needed during
projection. Dense tensors are reconstructed **one element at a time**, using
cached interpolation matrices and one compact coordinate/crop plan. Each output
XYZ/scalar/connectivity array is allocated once and populated directly, with
native VTK offsets/IDs instead of packed-cell concatenation. At most eight
element topologies are cached. VTK's borrowed NumPy buffers retain explicit
Python owners for the grid lifetime. Shared-face checks use bounded face
temporaries. The final output grid and the writer's complete read-back must
still fit in memory; this is not a streaming VTU writer.

Preflight uses native points `Ns`, target output points `Nt` and target cells `C`:

```text
B = 40*Nt + 80*C + 1 MiB
W = 1024*N^3 bytes for denser reconstruction; zero for the native path
RAM allowance = 8*V*Ns + source_bytes + 240*Ns + 6*B + W + 1 GiB
disk allowance = 2*B + 1 MiB
```

`B` is an array/output-size estimate, not a promised compressed VTU file size;
the larger disk allowance includes XML/base64/compression overhead. Estimates
use the full output domain even for a crop. For 26,112 elements, eight stored
variables and a roughly 0.399 GiB float32 source, the estimates are:

| Target nodes/axis | Element-local points | Hexes | B (GiB) | RAM allowance (GiB) | Disk allowance (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 native | 13,369,344 | 8,956,416 | 1.166 | 12.182 | 2.334 |
| 10 | 26,112,000 | 19,035,648 | 2.392 | 19.537 | 4.785 |
| 12 | 45,121,536 | 34,755,072 | 4.271 | 30.813 | 8.544 |

The bounded dense workspace allowance is about 0.977 MiB at 10 and 1.688 MiB
at 12, in addition to output buffers and native projection. These are formulas,
not measured peaks or reservations. Tens of millions of cells, per-element
interpolation/eigenvalue calls, full validation, compression, hashing, read-back
and viewer memory may be expensive. No real-data benchmark has been run.

A later visualization-convergence study should compare native/10/12 samples
for the same physical time, domain, camera, color/isosurface levels and field
definition. Evaluate differences at common physical coordinates using the P7
representation, distinguish scalar-linear viewer interpolation from the
tensor-first reconstruction, and quantify slice/contour/isosurface changes as
sampling increases. Keep the original mesh with no smoothing, subdivision,
resample-to-image or merging during comparisons. This is a proposed validation
procedure, not a convergence result or a claim of verified ParaView/Tecplot
import. Phase 3 batch processing and PVD generation remain deferred.

Phase 2 synthetic verification on 2026-11-07: **107 new reconstruction cases**;
**608 focused tests passed** (including the existing Phase 1 and numerical
regressions); **2,188 full-suite tests passed**, with zero failures/skips.
Compileall, CLI help and whitespace checks on tracked and untracked files
passed. Thirteen existing numerical/geometry/I-O modules remain byte-for-byte
identical to HEAD, including the GLL/refinement, derivative, projection,
vorticity, swirling-strength, slice and hex-validation kernels. The native
gradient/projection body, native mesh builder, crop checks and binary writer
are unchanged; only their orchestration was extended.

```bash
PYENV_VERSION=research312 python -m pytest -q tests/test_vortex_3d_reconstruction.py \
  tests/test_vortex_3d.py tests/test_vortex_3d_regressions.py \
  tests/test_vorticity.py tests/test_scalar_projection.py tests/test_l2_projection.py \
  tests/test_spectral_derivatives.py tests/test_swirling_strength.py -ra
PYENV_VERSION=research312 python -m pytest -q -ra
PYENV_VERSION=research312 PYTHONPYCACHEPREFIX=/tmp/nek-post-phase2-pycache \
  python -m compileall -q src scripts tests
PYENV_VERSION=research312 python -B scripts/40_export_vortex_3d.py --help
git diff --check
```
