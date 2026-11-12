# Single-frame 3D binary VTU: native grid and P7 reconstruction

The [numerical methods index](methods/README.md) provides the formulas, pinned
local Nek5000 correspondence, production provenance limits, and proposed
independent validation. This guide retains its existing workflow/API scope.

`scripts/40_export_vortex_3d.py` exports exactly one original snapshot as one
`vtkUnstructuredGrid`, with float64 PointData `omega_x` and `lambda_ci` on
the same physical points and cells. The default remains the Phase 1 native-grid
representation. Phase 2 adds optional element-local P7 reconstruction on denser
GLL nodal sets. The single-frame CLI has no batch, PVD, resume, overwrite or
smoothing option. [Phase 3](#phase-3-sequential-resumable-series) adds a separate
sequential series CLI with strict resume and a case-level PVD.
The Phase 1 description below still applies to the default path; the
[Phase 2 section](#phase-2-p7-volume-reconstruction) specifies denser sampling.

[Phase 4 compatibility diagnostics](vtu_compatibility.md) rewrite existing VTUs
with controlled VTK settings for manual Tecplot/ParaView testing. This separate
tool does not change the production writer or establish application compatibility.

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
import. The Phase 3 extension below adds temporal orchestration separately.

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

## Phase 3: sequential resumable series

`scripts/41_export_vortex_3d_series.py` calls the existing single-frame exporter
once per pending snapshot. It does not duplicate derivatives, projection,
interpolation, eigenvalues, cropping, mesh construction or binary serialization.
The default is `N7_H`, indices 1–81, native sampling and the positive-x volume
described below. Select explicit P7
sampling with `--sampling-nodes 8`, `10`, `12` or another integer >=8. The default
series parent root is `/data/Nek5000_data/results/h_refinement/vortex_3d_series`, separate
from the Phase 1/2 default roots. The single-frame CLI remains available.

### Relocated production series

The existing GLL10 N7_H series was relocated beneath `h_refinement` on
2026-11-12. Its frame-79 volume is available at

```text
/data/Nek5000_data/results/h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.vtu
```

`--output-dir` continues to select an external parent, with `gllN/CASE`
appended for dense sampling. The Python series API uses the same default.

PVD entries use paths relative to `series/`, so the collection travels with
the frame directories. Frame JSON sidecars store absolute `output_directory`
and `output_paths` values. Relocation therefore requires an explicit metadata
migration and updated per-frame metadata checksums in the active manifest.
The active processing summary also carries output and log locations.

The dataset-root `relocation_manifest.json` archives the exact original text
and SHA-256 of every migrated metadata file, the original file inventory and
checksums, and the source-to-destination mapping. Each migrated document links
to that record through `output_relocation`. The record also preserves the
original series-module source and both checkpoint signatures. The active
checkpoints were rebound only after verifying that the module's sole change
was the default output-root assignment and every other signature entry matched.
Frame numerical signatures identify the generating implementation. Historical
invocation logs retain their original paths, which the relocation record maps
to the current dataset. Workflow resume reads the active manifest, processing
summary and frame sidecars.

Strict resume continues to validate source hashes, element maps, configuration,
software provenance, VTU checksums and saved payloads before reusing frames.
For a read-only metadata/checksum preflight of the relocated series, run

```bash
PYENV_VERSION=research312 python -B scripts/41_export_vortex_3d_series.py \
  --indices 1-81 --sampling-nodes 10 --positive-x --resume --dry-run
```

The dry-run reports `validate_then_skip` for completed frames and defers VTU
payload decoding. The relocation audit separately exercises the saved-payload
validator without publishing a new production invocation.

### Complete positive-x volume

The intended N7_H production output is a **three-dimensional UnstructuredGrid**
covering `0 <= x <= 17`, `0 <= y <= 1.5`, `0 <= z <= 1`, containing both
float64 PointData arrays `omega_x` and `lambda_ci`. It is not a slice, surface
or isosurface dataset. `--positive-x` selects this preset explicitly; it is also
the default for cases named `N7_H`/`case_N7_H`. Other cases retain the previous
full-domain default. `--full-domain` explicitly requests all original cells,
including negative x. These flags and `--crop` are mutually exclusive.

Complete positive-side hexahedra are retained, including their **x=0 boundary
vertices**. No exported cell extends into x<0, and no strict `x>0` point mask
removes the first cell layer. The preset requires x=0 and x=17 to be exact
spectral-element boundaries, with complete native y=[0,1.5], z=[0,1] extents.
If the geometry does not satisfy these conditions, preflight reports the
limitation and execution stops; it does not clip cells, create edge samples,
adjust coordinates or silently discard part of the requested volume. No
production geometry/alignment has been inspected during implementation.

The series also recognizes an explicit `--crop 0 17 0 1.5 0 1` as this preset.
Other explicit crops retain the established complete-subcell semantics. The
single-frame CLI, its default full-domain selection and its arbitrary crop
behavior are unchanged, including its conservative header-only estimates.

Native gradients, curl and all nine-component mass-lumped projections still
use **the entire original source domain** before output selection. Negative-x
elements contribute to interface/periodic recovery, including values on x=0.
Only the output mesh and denser reconstruction are restricted to selected
elements. The dense builder does not allocate a full-domain high-resolution
grid and then discard its negative half. Both fields share the same native or
10/12-node sampled 3D mesh, with tensor-first lambda_ci unchanged. No ParaView
resampling or smoothing is introduced.

A compact coordinate/crop plan gives exact selected element-local point and
hex counts, actual output bounds and a full native-mesh signature. It is reused
across initial/locked preflight and subsequent frames; each frame's geometry is
checked before differentiation. Region policy, crop, counts and native geometry
are recorded in per-frame metadata, summaries and the manifest. Resume rejects
mixing full-domain and positive-volume results in one case output root.

`--indices` accepts unique comma-separated indices/ranges such as `1,41,79` or
`1-81`. Alternatively supply **both** `--start-index` and `--end-index`.
Omitting both forms selects 1–81. Duplicate/overlapping selections, reversed
ranges, booleans/nonintegers and indices outside the five-digit 1–99999 naming
range fail before calculation. `--max-frames N` caps the sorted selection after
validating the complete supplied selection, including duplicates. It limits
selected frames, not just newly computed frames. Nonconsecutive selections are
supported. Physical times must be finite, unique and strictly increasing with
frame index, including previously published frames retained in the collection.

### Output organization

For ten-node sampling:

```text
ROOT/gll10/case_N7_H/
  f00001/GC0_f00001_gll10.vtu
  f00001/GC0_f00001_gll10.json
  ...
  .vortex-3d-series.lock
  .vortex-3d-f00002-<unique>/       # preserved if an attempt is interrupted
  logs/<UTC>_<unique>.jsonl
  logs/<UTC>_<unique>_summary.json
  series/case_N7_H_vortex_3d_series.pvd
  series/manifest.json
  series/processing_summary.json
```

Native output omits `gll10/` and the `_gll10` filename suffix. Explicit eight-node
sampling uses native filenames, as in the single-frame workflow. Each sampling
density has its own case root and lock. Frame artifacts retain the existing
names and immutable no-replace directory publication. The case-level collection
is placed in `CASE/series/` so scientific frame ancestry checks remain strict;
it references `../f00001/GC0_f00001_gll10.vtu`, never private staging.

The PVD may represent a partial/nonconsecutive selection. Every update includes
**all validated completed frames in that case root**, preserving prior entries
and sorting by actual physical timestamps. A successful `completed` invocation
means its requested selection completed; inspect `selected_indices`,
`selected_frames` and `published_frames` before interpreting it as a complete
81-frame run. PVD timestamps never come from an index or assumed interval.

### Preflight, resume and publication

Preflight validates every selected header and original one-based element map,
as well as existing published frames needed to preserve the collection. It
checks paths, actual times, configuration, producer/source checksums, software
versions, signatures, resources and PVD references. Both original snapshots and
generated scientific files must remain outside the repository. Symlinked
output ancestry, control files, frames and staging directories are rejected.
Unknown scientific frame/control artifacts are preserved and rejected.

`--dry-run` is read-only: no lock, directory, log, checkpoint, velocity payload,
VTK decoder, derivative, eigensolve or reconstructed 3D mesh is created. It reads
headers/maps, JSON/XML metadata and **streams source/VTU checksums and binary XML
headers**. For a cropped output it also reads one source's **native coordinates
only**, behind a RAM gate, to validate boundaries and count selected cells.
Velocity/pressure/temperature/scalar fields are skipped. This planning uses
the existing structured geometry and Phase 2 GLL plan, not a high-resolution
3D mesh. The RAM gate also includes the per-element dense workspace, rejecting
unsafe target counts before target GLL-node generation. It can still read
substantial bytes: a full source inventory is
not a cheap directory listing. Existing outputs are reported as
`metadata_valid` with `payload_validation=pending` and action
`validate_then_skip`; dry-run never labels an undecoded payload fully valid.
Missing/incompatible inputs and resource failures are reported in `failures`
with `status=invalid`. Invalid argument combinations/unsafe control locations
can raise immediately. The CLI returns failure for an invalid plan.

Execution obtains the existing persistent advisory-lock helper, then repeats
preflight under that lock. Sources/signatures must match the initial plan.
Actions and resources are refreshed, so another cooperating writer's newly
completed frame is validated and reused rather than recomputed. Locks are
released on exceptions; their inode is never removed, and stale PID text is
not treated as ownership. All writers must honor the lock; resource checks
are not RAM/disk reservations against other jobs.

Without `--resume`, any requested existing frame stops the run. With resume,
every reused frame must have its exact VTU/JSON pair, completed metadata,
matching source/header/time/configuration, current producer and series
signatures, current recorded Python/library versions (including SciPy's GLL
node dependency), source/VTU checksums and
valid binary format. A VTK reader then checks actual float64 PointData,
`TIME_VALUE`, ranges, geometry, connectivity, cell validity and all recorded
array fingerprints. One saved grid is decoded at a time after a RAM gate.
Metadata hashes are also checked against the case manifest when present.
An incompatible or corrupted final frame is never silently recomputed over.

Artifact identities use `lstat()` and include device, inode, regular-file mode,
size, modification time and change time. Both VTU and JSON must be nonempty
regular files. Reads use `O_NOFOLLOW`, `O_NONBLOCK` and descriptor-based identity
checks before/after reading, together with a check of the named entry. Expected
identities are checked before JSON parsing, VTU hashing or payload decoding;
the Linux VTK reader uses the pinned descriptor through `/proc/self/fd/`.
Recovery rechecks staged identities immediately before no-replace publication.
The single-frame producer also rechecks both newly staged files before publication.
Reused artifacts are checked during execution, before PVD publication and
before the final completion checkpoint. A same-content symlink is never an
acceptable substitute for the validated regular file.

The advisory lock serializes cooperating writers; it cannot prevent a process
with write access from changing directory entries after the last check. Linux
has no operation that atomically verifies a directory's contents and renames
it. If an uncooperative mutation occurs inside that final rename gap, the
post-publication checks fail, no PVD/checkpoint declares the frame completed,
and any already-renamed invalid final directory is retained for diagnosis.
Strict resume rejects that directory; it is never deleted or overwritten.
Likewise, existing PVDs cannot be protected from someone subsequently replacing
their referenced files outside the locking protocol. Keep output directories
under controlled write access; these checks are not a filesystem snapshot.

New frame metadata adds bounded SHA-256 fingerprints of coordinates, native
connectivity/offsets/types, the two scalar arrays and time. Full original
native geometry/order signatures cover **every source element, including areas
outside a crop**. Subsequent frames must match before differentiation, and
all saved output meshes must match. File-order permutations in valid Nek maps
remain distinct in source provenance while the reader's correct global element
ordering is retained. No cross-frame scientific arrays or VTK objects are
cached. Small interpolation matrices retain their existing process-local cache;
native operators and geometry plans are rebuilt by the same single-frame API.

Existing Phase 1/2 outputs remain protected. Historical files without these
fingerprints or with earlier implementation signatures are not automatically
adopted, even under `--resume`; use a fresh output root rather than overwriting
them. No broad legacy-signature exceptions are introduced.

Explicit eight-node native metadata now states that dense interpolation and
the reconstruction-route omega audit were not performed, addressing the
earlier provenance wording issue. The audit flag stays false during dry-run
and becomes true only after actual denser reconstruction. Scientific arrays
and the native calculation path are unchanged.

For each generated frame, the producer writes only inside a private sibling
directory, validates the mesh, saves binary VTU, verifies exact read-back, checks
inline binary format, fingerprints arrays and rechecks source/implementation
integrity. Only then does Linux `renameat2(RENAME_NOREPLACE)` publish the pair.
The series reuses that successful read-back rather than decoding the same new
grid again. Frame-local allocations are released before the next frame.

PVD publication uses a unique same-directory temporary, flush/fsync, XML
read-back and exact manifest/reference checks. Initial creation uses atomic
no-replace rename; later updates atomically replace only the validated control
collection. The old PVD remains intact if staging/replacement fails. The case
lock and unchanged-file checks guard cooperating updates. Operational JSON
checkpoints use the existing atomic helper, and invocation logs use exclusive
new names. Final verification rehashes sources and immutable frame artifacts.
This is crash-safe publication, not a filesystem snapshot or a power-loss
durability guarantee.

### Interruption and recovery

Final frame directories are the recovery ground truth; progress logs are
diagnostics. If a frame was published but PVD/manifest/checkpoint updates were
interrupted, `--resume` validates the orphan final frame and reconciles the
collection/checkpoint without recalculation. Existing PVD entries pointing to
missing, invalid, duplicate or wrong-time frames fail closed.

Unpublished staging blocks normal execution of that pending frame. Explicit
`--resume --recover-staging` validates a uniquely complete staged pair before
atomic publication. Recovery needs no new scientific-output disk allocation.
Unreadable or unfinished staging is preserved and a new private attempt is
used; no old staging is deleted. Incompatible completed staging or multiple
complete candidates stops the run for manual assessment. An incomplete or
corrupted **final** directory is never replaced by recovery. Preserve it and
investigate, or choose a separate new output root. Do not delete lock files
while a writer might be active.

### Series resources

For series crops, exact selected counts replace full-domain **output** counts
in the Phase 1/2 allocation formulas. Input/native node counts, full-domain
gradient/projection memory and the bounded per-element reconstruction allowance
remain unchanged. If `Ns` is native source points and `Ps,Cs` are selected
element-local output points/cells:

```text
Bselected = 40*Ps + 80*Cs + 1 MiB
RAM = 8*V*Ns + source_bytes + 240*Ns + 6*Bselected + W + 1 GiB
D = 2*Bselected + 1 MiB
```

The series does not assume that half the physical domain contains half the
elements: nonuniform partitions are counted from actual coordinates. These
output-size formulas are allowances, not measured compressed file sizes.
If `D_i` is each pending
new frame's conservative single-frame disk allowance, series preflight reserves:

```text
sum(D_i) + max(D_i, default=0) + 64 MiB
```

The control allowance grows beyond this minimum for unusually large frame
inventories; metadata/summary RAM is also included. The 81-frame figures below
use the 64-MiB minimum. JSON metadata is bounded at 64 MiB; inventories whose
conservative control-file bound exceeds that budget fail before header I/O.

The extra frame allowance covers staging/headroom; read-back is an additional
RAM cost rather than a duplicate disk file. Already published or recoverable
frames add no new scientific disk reservation. Preserved incomplete staging
already consumes current free space and is never assumed reclaimable. Remaining
disk is checked again before every frame. Calculation RAM is the maximum
single-frame allowance, not its sum over time. Saved-frame validation uses
`4*max(estimated_output_bytes, actual_vtu_bytes) + 1 GiB` and holds one grid.

For reference, all 81 **full-domain** frames with the earlier float32/eight-variable assumptions:

| Nodes/axis | RAM allowance | Per-frame disk allowance | Initial series disk allowance |
| --- | ---: | ---: | ---: |
| 8 | 12.182 GiB | 2.334 GiB | 191.420 GiB |
| 10 | 19.537 GiB | 4.785 GiB | 392.429 GiB |
| 12 | 30.813 GiB | 8.544 GiB | 700.642 GiB |

As an illustration only, if coordinate planning finds exactly 13,056 selected
elements out of 26,112, the positive-volume allowances would be:

| Nodes/axis | Selected points | Selected hexes | RAM | Per-frame disk | 81-frame initial disk |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 6,684,672 | 4,478,208 | 8.686 GiB | 1.168 GiB | 95.862 GiB |
| 10 | 13,056,000 | 9,517,824 | 12.364 GiB | 2.394 GiB | 196.366 GiB |
| 12 | 22,560,768 | 17,377,536 | 18.002 GiB | 4.273 GiB | 350.472 GiB |

This **does not assert** the actual N7_H positive element count. That count and
alignment must come from an authorized coordinate-only preflight. Native
projection in this example still uses all 13,369,344 original element-local
GLL points; it is not a positive-only recovery.

These are unbenchmarked conservative allowances, not measured compressed file
sizes. Crops reduce output allowances but retain full-domain native projection.
Preflight checks the process soft `RLIMIT_FSIZE` and detectable FAT-family
four-GiB file limits. Other filesystem maximums, quotas, compression ratios,
viewer memory and runtime remain unknown and are reported as uncertainty.
Hashing/decoding are linear passes over scientific files; progress updates do
not rehash every previous large VTU on every frame. Geometry validation is
never disabled. No real-data benchmark or series has been run for this feature.

### Proposed commands (not executed on real data)

Read-only preflight for a fresh positive-volume series destination:

```bash
PYENV_VERSION=research312 python -B scripts/41_export_vortex_3d_series.py \
  --case N7_H --indices 1-81 --sampling-nodes 10 --positive-x \
  --output-dir /data/Nek5000_data/results/vortex_3d_positive_x_phase3_production_new \
  --periodic-axes x y --expected-element-counts 272 12 8 --dry-run
```

Selected-frame smoke test, requiring separate authorization for real data:

```bash
PYENV_VERSION=research312 python -B scripts/41_export_vortex_3d_series.py \
  --case N7_H --indices 1,41,79 --sampling-nodes 10 --positive-x \
  --output-dir /data/Nek5000_data/results/vortex_3d_positive_x_phase3_smoke_new \
  --periodic-axes x y --expected-element-counts 272 12 8
```

Full 81-frame production, only after reviewing the dry-run and authorizing it:

```bash
PYENV_VERSION=research312 python -B scripts/41_export_vortex_3d_series.py \
  --case N7_H --start-index 1 --end-index 81 --sampling-nodes 10 --positive-x \
  --output-dir /data/Nek5000_data/results/vortex_3d_positive_x_phase3_production_new \
  --periodic-axes x y --expected-element-counts 272 12 8
```

Resume/recover the same configuration after interruption; adding `--dry-run`
first gives a read-only plan and defers VTU payload validation:

```bash
PYENV_VERSION=research312 python -B scripts/41_export_vortex_3d_series.py \
  --case N7_H --indices 1-81 --sampling-nodes 10 --positive-x \
  --output-dir /data/Nek5000_data/results/vortex_3d_positive_x_phase3_production_new \
  --periodic-axes x y --expected-element-counts 272 12 8 \
  --resume --recover-staging
```

For a future ParaView session, open the case's `series/*_vortex_3d_series.pvd`,
Apply, select PointData `omega_x` or `lambda_ci` and use its actual time steps.
Keep consistent camera/ranges and the original unmerged mesh; no smoothing,
subdivision or resample-to-image is needed to recover exported samples.
The PVD supplies time navigation, not higher-order spatial interpolation.
ParaView and Tecplot imports/rendering remain unverified.

### Phase 3 synthetic verification

Continuation verification on 2026-11-08: **781 focused tests passed in 34.82 s**;
the complete suite passed **2,361 tests in 89.83 s**, with zero failures/skips.
The focused suite includes 565 exporter cases. Phase 3 contributes
120 series cases and 53 positive-volume cases, including interruption/recovery,
corruption, no-replace publication, actual-time PVD, crop-policy resume checks,
all six physical-axis permutations with reversed elements, native/8/10/12
sampling, exact cropped/full point/field/connectivity comparisons, x=0 cell
layers and whole-domain projection contributions. Coordinate-only dry-runs and
per-frame reconstruction scopes are checked independently with call guards.
Large target counts fail before coordinate/interpolation planning when the
bounded dense workspace cannot fit; crop-adjusted resource-status flags are
also checked against their revised RAM/disk budgets.

Compileall, both CLI help checks, `git diff --check`, and whitespace checks of
all twelve modified/untracked text files passed. Seventeen existing numerical,
geometry and I/O modules and scripts 35/39/40 match HEAD byte for byte. The
native gradient/projection arithmetic, crop, native mesh builder and binary
writer bodies also match HEAD. No real CFD dataset, production job, viewer or
real-data benchmark was run. Resource figures remain estimates; the real case's
positive-boundary alignment is deliberately checked by a later authorized
preflight rather than assumed from its name.

```bash
PYENV_VERSION=research312 python -m pytest -q \
  tests/test_vortex_3d_positive_volume.py tests/test_vortex_3d_series.py \
  tests/test_vortex_3d.py tests/test_vortex_3d_regressions.py \
  tests/test_vortex_3d_reconstruction.py tests/test_vorticity.py \
  tests/test_scalar_projection.py tests/test_l2_projection.py \
  tests/test_spectral_derivatives.py tests/test_swirling_strength.py -ra
PYENV_VERSION=research312 python -m pytest -q -ra
PYENV_VERSION=research312 PYTHONPYCACHEPREFIX=/tmp/nek-post-phase3-positive-volume-pycache \
  python -m compileall -q src scripts tests
PYENV_VERSION=research312 python -B scripts/40_export_vortex_3d.py --help
PYENV_VERSION=research312 python -B scripts/41_export_vortex_3d_series.py --help
git diff --check
```

Artifact-integrity correction verification on 2026-11-08: **55 new regression
cases passed in 10.45 s**; **228 Phase 3 tests passed in 39.19 s**;
**608 Phase 1/2 and numerical regressions passed in 8.38 s**; the complete suite
passed **2,416 tests in 101.67 s**, with zero failures/skips. Tests cover both
artifact types, same-content and copied-target symlinks, missing/empty files,
directories/FIFOs, regular-file replacement, no-follow open races, pinned VTK
reads, descriptor cleanup, new/recovered staging and PVD publication boundaries.
The three original reproductions now abort without completing the invocation;
staged substitutions are rejected before publication and original bytes/PVDs
are preserved. The explicit uncooperative-rename-gap test checks the limitation
described above: fail without a PVD update or completed invocation, preserving
any final directory that the kernel already renamed.

Compileall, both CLI help checks, `git diff --check` and whitespace checks of
all thirteen modified/untracked text files passed. Scientific kernels,
selection/reconstruction, connectivity, float64 fields and physical time are
unchanged. No real CFD data or viewer validation was required or run.

```bash
PYENV_VERSION=research312 python -m pytest -q \
  tests/test_vortex_3d_artifact_integrity.py tests/test_vortex_3d_series.py \
  tests/test_vortex_3d_positive_volume.py -ra
```
