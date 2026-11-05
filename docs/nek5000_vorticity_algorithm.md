# Native-GLL streamwise vorticity and scalar L2 recovery

## Definition and discrete sequence

For Cartesian velocity `(u,v,w)`, `omega_x = dw/dy - dv/dz` (inverse time).
The calculation preserves the sign. It uses the complete original 3D domain
at native GLL nodes; no cropping, slicing, smoothing, thresholding, high-order
volume interpolation, other curl components, or eigenvalues are involved.

The sequence is native velocity -> local reference derivatives -> physical
derivatives -> local omega_x -> mass-lumped scalar projection -> native VTR.
Computing the curl before projection follows Nek5000's operation order.

## Nek5000 correspondence

The official [core/navier5.f](https://github.com/Nek5000/Nek5000/blob/master/core/navier5.f)
was inspected on 2026-10-16, specifically `local_grad3` and `comp_vort3`.
`local_grad3` applies one-dimensional differentiation matrices along the three
reference directions. In 3D, `comp_vort3` transforms these derivatives with
geometry metrics and the inverse Jacobian, forms the local curl, and applies
`bm1`, `dssum`, then `binvm1` to each curl component.
The assembly routine is in [core/dssum.f](https://github.com/Nek5000/Nek5000/blob/master/core/dssum.f).
Only the first curl component is needed here.

## Differentiation and physical coordinates

`spectral_derivatives.reference_derivatives` retains the existing float64 GLL
differentiation matrices. Element arrays use `(nt,ns,nr)`; reference `(r,s,t)`
corresponds to array axes `(2,1,0)`.

`element_jacobian` is unchanged: `J[i,j] = dx_i/d(reference_j)`. It validates
finite geometry, conditioning and determinant sign. `vorticity.py` solves
`J*c_y = e_y` and `J*c_z = e_z` to obtain only the two required inverse columns.
Their dot products with the reference derivatives of `w` and `v` yield
`dw/dy` and `dv/dz`. The difference is formed immediately for each element.
No complete physical gradient of either field, full nine-component tensor,
or full-volume pair of derivative arrays is allocated.

## Shared scalar projection

For the existing `L2ProjectionOperator`, local mass is
`m[e,a] = abs(detJ[e,a]) * wt[a_t] * ws[a_s] * wr[a_r]`.
For each structured global node `I`, scalar recovery is

```text
P(g)[I] = sum_(e,a -> I) m[e,a] * g[e,a] / sum_(e,a -> I) m[e,a]
```

`project_scalar_field` accepts real finite `(nelem,nt,ns,nr)` arrays and returns
float64 values in the same local layout. It preserves signs, supports chunked
processing and optional non-overlapping contiguous output storage, and can
validate a supplied mesh before operator reuse. The existing explicit periodic
x/y mapping participates in assembly. The assembled RHS uses one global vector.

`project_velocity_gradient` retains its public API, validation, output rules
and nine-component diagnostics. It invokes the same internal scalar kernel
nine times while reusing the RHS. Contribution order, multiplication dtype,
mass division and gather order match the pre-refactor implementation.

The existing lambda_ci sequence remains full local gradient -> nine-component
mass-lumped projection -> eigenvalues. Dense slice workflows still interpolate
all nine projected components in P7 before evaluating eigenvalues. No existing
CLI or scientific field format is changed by this refactor.

## Native signed volume export

The exact rectilinear coordinate assembly was extracted from
`lambda_ci_export.py` into `rectilinear_export.py` and is shared by both exports.
It checks original coordinate replicas and adjacent endpoints exactly, detects
physical/array-axis permutations and element orientation from the prepared map,
and retains both physical endpoints of periodic directions. Scalar shared nodes
must agree exactly; nothing is averaged or interpolated for visualization.
The lambda_ci-specific nonnegative and squared-field checks remain separate.

The omega_x VTR contains only float64 PointData `omega_x`, physical x/y/z vectors,
x-fast ordering `(Iz*Ny+Iy)*Nx+Ix`, and actual snapshot `TIME_VALUE`. N7_H has
26,112 elements, P7 (8 nodes per edge), projection shape `(1904,84,57)`, and
visualization shape `(1905,85,57)` including periodic endpoints. VTR cells are
linear cells between native nonuniform GLL nodes, not native high-order elements.

## CLI and output safety

```bash
PYENV_VERSION=research312 python -B scripts/36_compute_omega_x.py \
  --case N7_H --indices 79 --dry-run
PYENV_VERSION=research312 python -B scripts/36_compute_omega_x.py \
  --case N7_H --indices 79
PYENV_VERSION=research312 python -B scripts/36_compute_omega_x.py \
  --case N7_H --indices 79 --resume
```

`--case` accepts `N7_H`, `case_N7_H`, or a case directory path. Default selection
is frame 79 only. `--indices` accepts comma-separated integers/ranges; alternatively
supply both `--start-index` and `--end-index`. `--output-dir` is an output root,
defaulting to `/data/Nek5000_data/results/h_refinement/omega_x`. Results live in
`ROOT/CASE/fNNNNN/CASE_fNNNNN_omega_x.{vtr,pvd,json}`.

Dry-run reads headers/resources and validates any requested completed results;
it creates no files. Missing inputs, incompatible headers, resource limits,
unsafe paths, and output collisions stop processing. A case advisory lock
serializes cooperating workflow writers. Each new frame is written in a unique
private `.omega-x-staging/` directory, checked with the VTK reader against the
in-memory coordinates and values, and atomically published without replacement.
Metadata records source/output SHA-256, implementation signatures, scalar range,
dimensions, coordinate hashes, stage runtimes and process peak RSS.

### RAM preflight and sequential resume validation

Discovery reads snapshot headers and output file stats without constructing a
VTK reader. It estimates the entire requested batch before source/output
checksums or volume validation. Insufficient or unknown available RAM returns
`resources_acceptable=false` with explicit `resource_errors` in dry-run; no
volume reader or native computation is invoked. Normal execution raises
`MemoryError` before writing outputs. A reused frame remains marked
`resume_validation=pending` until its full validation succeeds.

For this workflow's uncompressed float64 single-field VTRs, let `N=Nx*Ny*Nz`,
`V=8*(N+Nx+Ny+Nz)` bytes for scalar/coordinate arrays, and
`B=max(V,VTR_file_bytes)`. For new staging, the existing output-size estimate
is used until the actual file size is available. The validation budget is

```text
M_validation = 4*B + R + 1 GiB
R = retained operator bytes (zero until a frame requires computation)
R += bytes of the native field and coordinates for staging read-back
M_frame = M_validation                             (reused frame)
M_frame = max(M_computation, M_validation)          (new/overwrite frame)
M_batch = max(M_frame over requested frames)
```

The four-payload allowance covers file/decode buffers, VTK dataset arrays,
NumPy conversion and validation scratch. `vtk_to_numpy` often returns views;
the conservative estimate does not require zero-copy behavior. The existing
1-GiB margin covers bounded checksum buffers, library overhead and uncertainty.
The original computation estimate includes reader/source buffers, local raw
and projected scalars, local mass weights, assembled mass/RHS, visualization
storage and its original safety margin. The cache extension below adds mapping
overhead and mesh-validation scratch. The batch takes a maximum because
frames are processed and released sequentially, not held together in memory.

The existing host/cgroup availability probe is reused. Probe errors or invalid
results (including unknown, noninteger, negative or nonfinite values) fail
closed. A fresh availability check precedes each source read/computation and
each full VTR validation, including direct validator calls and staging
read-back while its native input remains retained. The checks precede VTK
reader construction and `Update`, coordinate/PointData extraction and scalar
comparison. A late RAM drop during resume preflight returns an unsafe report.
Memory is checked at these boundaries; it is not reserved against competing
processes, and power-loss/allocator behavior is not guaranteed by the estimate.

`--resume` validates completed source/output checksums, configuration, signatures,
VTR topology/coordinates/scalars/time and the single-frame PVD before reuse.
An incomplete or incompatible final directory is never silently skipped.
Before operator caching, the RAM fix permitted one explicitly reviewed omega_x workflow transition:
the predecessor SHA-256
`9f261baadf8e888ef216f83dc5cdcd2f9a4336e33c5b200340c60e22551d3736`
may be reused only with the exact reviewed RAM-fix successor source, whose
full-file SHA-256 is
`91e50314111ed4a4048ab445af41be57d7268eacdd1066251e74ee42434f713f`.
The embedded fingerprint hashes every source byte after normalizing
only that fingerprint's own 64-hex literal to zeros; the current full-file
digest must also match the source being checked. Any later workflow edit
invalidates this exception. All other numerical-module hashes and signature
keys must match exactly, and every source/configuration/checksum/VTR/PVD check
still runs. New results continue to record full-file SHA-256 signatures.
Historical metadata is not rewritten and no compatibility manifest is added.

`--overwrite` is explicit and mutually exclusive with resume. It accepts only
recognized omega_x directories containing the three expected files, preserves
the old frame in a unique `.omega-x-backups/` directory, and restores it if new
publication fails. Unknown files/symlinks stop overwrite. No existing lambda_ci
or original simulation output is an eligible destination.

## Tests and validation

The dated results below describe successive historical development stages, not
the current test count. The self-contained three-frame resume test and current
P7 review commands are documented in
[omega_x_p7_midplane_slice.md](omega_x_p7_midplane_slice.md#pytest-skip-investigation-and-resolution).

Focused tests cover signed constants, nonuniform shared-node mass weighting,
periodic x/y faces/edges/corners, independent coordinate-grouped mass assembly,
projection linearity, solid rotation, shear, polynomial curl, skew/curved local
Jacobians, axis permutations, reversed elements, VTR round trips, resume,
overwrite backups/rollback, collision protection and resource failures.
Frozen pre-refactor arithmetic checks nine projected components and lambda_ci
exactly for signed random float32/float64/integer noncontiguous tensors.

Validation on 2026-10-16 in `PYENV_VERSION=research312`:

```bash
python -B -m pytest -q tests/test_scalar_projection.py tests/test_vorticity.py \
  tests/test_omega_x_workflow.py tests/test_l2_projection.py \
  tests/test_lambda_ci_workflow.py -p no:cacheprovider
python -B -m pytest -q -p no:cacheprovider
git diff --check
```

Focused tests: **141 passed**. Complete suite: **1605 passed in 28.05 s**,
including all 1554 prior tests and 51 added cases. Whitespace checks passed for
tracked and untracked files. The frozen legacy projection and lambda_ci scalar
arrays matched exactly in the numerical regressions. An additional pre-refactor
export comparison produced identical physical coordinates, scalar arrays and
VTR file bytes on a small nonuniform P7 mesh.

Header/resource dry runs passed before production. Frame 79 was generated and
validated first; only after its successful VTR/grid comparison were the nearest
actual-time frames to `t=10` and `t=13.5` selected and processed:

| Frame | Actual time | omega_x min | omega_x max | Frame elapsed (s) | Process peak RSS (GiB) |
| --- | --- | --- | --- | --- | --- |
| 79 | 19.50240408807 | -37.8430021494745 | 34.34511365452145 | 58.68 | 1.075 |
| 41 | 10.0035071079 | -17.850693327316716 | 14.92993859514711 | 57.26 | 1.074 |
| 55 | 13.50377813401 | -21.316594616520387 | 15.733644929486516 | 56.28 | 1.122 |

Frame elapsed includes native calculation, export, read-back and publication;
CLI totals including input hashes/preflight were 60.75 s for frame 79 and
117.62 s for the 41/55 batch. RSS is a process high-water mark, cumulative
within the two-frame process, not a per-frame isolated allocation measurement.

Each frame has 9,229,725 finite float64 points with both signs. VTR dimensions
are `(1905,85,57)` and bounds are `[-17,17] x [0,1.5] x [0,1]`. All three
coordinate arrays match the existing full-volume frame-79 lambda_ci VTR exactly;
that corresponding reference was available. Frame-79 time and x-fast first-cell
connectivity also match. Every generated VTR was compared against its in-memory
native coordinates and signed values before publication. Source hashes were
verified before/after calculation and again in the final read-only audit.

Outputs are in
`/data/Nek5000_data/results/h_refinement/omega_x/case_N7_H/`, under
`f00041`, `f00055`, `f00079`, each containing its matching VTR/PVD/JSON manifest.
No other production frames were requested. A final `--resume --dry-run` on all
three validated their completed records without modifying output files.
All 676 pre-existing lambda_ci files retained their original SHA-256 hashes.

The subsequent RAM-preflight fix adds 23 regression cases: 17 cover low/unknown
RAM, mixed/new/reused frames, nonzero sequential peak estimates, reader spies,
late availability drops and retained-field staging read-back; six cover the
precise predecessor exception, rejection of other software signatures and
continued checksum enforcement. Before the fix, the four parameterized
low-RAM resume cases failed because the VTR reader was invoked before resource
rejection. Focused tests now pass **164 cases** using the command above.
The complete suite passes **1628 tests in 27.21 s** (1605 existing plus 23
new cases); tracked and all eight untracked files pass whitespace checks.
Real `--resume --dry-run` on frames 41/55/79 passes without native payload
loading or recomputation. Its sequential validation peak is **1,369,164,124
bytes** (about 1.275 GiB), with zero computation RAM required. Forcing available
RAM to one byte returns an unsafe dry-run report and raises `MemoryError` in
normal execution, with **zero VTR reader calls**. A full inventory/SHA-256 audit
confirms all **686 existing files** (10 omega_x, 676 lambda_ci) are unchanged.

A follow-up compatibility audit adds 16 cases, bringing the focused command
to **180 passed** and the complete suite to **1644 passed in 29.80 s**. It
individually rejects changes to all five non-workflow signature modules,
periodicity/element-count/order settings, source hashes, timestamps, and VTR/PVD
checksums. VTR coordinate, timestamp, scalar and topology validation also reject
altered synthetic files even after their fixture checksum is refreshed. At that
stage, an optional read-only integration test validated historical frames
41/55/79 and checked RAM before all three reader constructions. Those scratch
outputs were subsequently deleted. The current deterministic replacement uses
only pytest's `tmp_path` and retains the resume, timestamp, no-recomputation,
RAM, VTR read-back and checksum assertions; see the P7 document linked above.

Normal execution (`resume=True`, not dry-run) also completed for all three
historical frames without native velocity loading, projection preparation,
recomputation or export. All nine VTR reads across initial preflight,
under-lock revalidation and final reuse were preceded by RAM checks. The
existing writer lock required access outside the read-only sandbox; no output
or metadata contents were changed.

The exact predecessor source and reviewed successor have identical ASTs for
scientific computation and export calls. All five other module SHA-256 values
match every historical frame. A two-frame synthetic P7 old/new workflow
comparison additionally verifies exact equality of complete float64 scalar
arrays, coordinates, TIME_VALUE, topology, projection configuration, source
hashes and PVD bytes. Freshly serialized VTR byte hashes can differ because
VTK assigns unnamed coordinate-array labels containing object addresses;
existing production files are never reserialized during resume and retain
their original SHA-256 checksums.

## Limits and equivalence claims

This implements the same mathematical mass-lumped recovery procedure, not
bitwise Nek5000 equivalence. Python reconstructs metrics from stored snapshot
coordinates (N7_H stores float32), uses float64 NumPy differentiation/linear
solves and a serial structured scatter. Nek5000 uses its internal geometry,
compiled matrix products and gather/scatter ordering. Our normalized inverse
columns also distribute the Jacobian scaling differently in floating point.
No direct Nek5000-generated vorticity reference field has been supplied.

Global assembly remains restricted to complete affine axis-aligned structured
boxes with explicitly confirmed translational periodicity. Local curl supports
valid more general geometry, but the workflow does not support curved/unstructured
global meshes, rotational periodicity or nonconforming interfaces.

Existing high-order lambda_ci results are not rewritten. Their strict resume
guard hashes the complete `l2_projection.py` file, so this refactor intentionally
triggers an implementation-signature mismatch for earlier completed metadata.
Numerical regression equality does not bypass that provenance protection;
resuming old series still requires a separate explicit compatibility review.
Appending to a PVD that includes those older completed frames also invokes
their validation and is subject to the same limitation; a new empty output
root avoids reusing historical records. The slice-series implementation and
its guard remain unchanged. The narrow omega_x RAM-fix exception above does
not apply to lambda_ci.

The initial RAM-only fix rebuilt projection operators per new frame. The
subsequent invocation-local cache described below removes that repeated work;
it does not introduce a persistent cache.

## Invocation-local projection operator reuse

`run_omega_x_series` starts with no operator. It builds one for the first
frame requiring computation, retains it through later frame reads and VTR
read-back, and releases it when the invocation exits. Reused completed frames
do not cause construction; an entirely completed resume builds zero operators.
There is no disk serialization, global cache or reuse across invocations.
New metadata records `operator_build_count`, `operator_reused`,
`operator_storage_bytes` and separate preparation/mesh-validation stage times.

The prepared node map, element GLL mass weights and assembled diagonal mass
depend on geometry, node ordering and explicit periodic configuration, not
velocity. Previously every frame repeated node-map construction and
`element_gll_weights` for all 26,112 elements, then assembled mass again.
The first uncached benchmark measured 8.73-8.79 s for mapping, 19.53-19.82 s
for element weights (GLL coordinate derivatives, Jacobian validation and
quadrature), and 28.38-28.75 s for total operator construction.

Before each numerical use, the workflow checks periodic axes, element interval
counts, polynomial/node shape and global node dimensions. The existing
`validate_mesh` compares coordinate signatures for every element in sequence:
all three full coordinate arrays and their shape must agree exactly after
the established float64 canonicalization. It accepts no coordinate tolerance
for reuse. Changed geometry, element order, local orientation or array-axis
layout is rejected. These exact coordinates preserve the original physical
axis mapping, node IDs, shared/periodic connectivity and geometry weights.
Mismatch raises an error; the workflow never silently rebuilds and continues.
The independent validation already inside curl computation and visualization
assembly remains in place. Only one operator is retained.

GLL differentiation, Jacobian transformations, signed `dw/dy-dv/dz`, scalar
mass-lumped projection and export kernels are unchanged. The new cache tests
compare fresh and cached weights, mass and complete float64 projected arrays
exactly, including a second snapshot with different velocities. No smoothing,
interpolation or arithmetic reordering is introduced.

### Retained memory

For `L` local nodes, `G` periodic global nodes, `E` elements and native
visualization dimensions `Nx,Ny,Nz`, the operator budget is

```text
C = 8*(L + G + Nx + Ny + Nz) + E*(3*8 + 3 + 256)
S = 8*E + 4*8*(nodes per element)
```

`C` covers float64 weights/mass, int64 element cells, boolean orientations,
physical coordinate vectors and 256 bytes/element for Python signature and
mapping overhead. `S` covers an element-reference tuple plus one element's
three float64 coordinate conversions and contiguous hash buffer. The existing
computation estimate already includes weights and mass; it adds only the
remaining `C` overhead and `S`, without counting a second operator. Actual
prepared storage plus the signature allowance is checked against this budget.

After the first computed frame, VTR validation adds `C` to retained bytes,
including completed frames interleaved with new ones. Staging adds both the
native field/coordinates and the retained operator. Entirely reused batches
retain no operator. Batch RAM remains the maximum of sequential frame peaks.
The existing 1-GiB safety margin, host/cgroup probe, unknown-memory rejection,
pre-reader gates and live availability checks are preserved. The cache's mesh
validator also checks availability before its temporary allocation. Estimates
are conservative; runtime available memory already reflects retained arrays,
so repeated gates may reject earlier rather than reserve memory optimistically.

### Cache-version provenance

Legacy compatibility is restricted to the two exact historical workflow
digests listed above (`9f261b...` and `91e503...`) and the specifically reviewed
cache successor, full-file SHA-256
`8b7b441f7eb45d7deeb3159b98d256502231b0377151ae1fa7214997241d6176`
(normalized successor fingerprint
`3b68a0982eb3dab6d3f15f83dc3770fce2a854e9b59007110f77fb163575e42f`).
All five other signature modules must still match byte for
byte. The full-file/normalized-successor check includes the guard itself;
unreviewed future source edits disable this legacy exception. Current results
record the actual full-file SHA-256. Configuration, source hashes, VTR/PVD
checksums and physical-data validation are not bypassed. No historical metadata
or production output is rewritten. The lambda_ci signature guard is unchanged.

### N7_H benchmark and real-data verification

The uncached baseline is the exact committed RAM-fix workflow (`91e503...`).
It and the final cache successor were run in separate Python processes on
the same frames 41/55/79 with identical native-grid, chunk-size, periodic and
output settings. The historical calculation outputs were stored under
`/tmp/omega-x-operator-cache-review/uncached-outputs/` and
`/tmp/omega-x-operator-cache-review/cached-final-outputs/`, with a preliminary
cached measurement in `cached-outputs/`. These temporary directories are no
longer present as of the 2026-11-05 review; the table is a historical record,
not evidence that can currently be inspected at those paths. It uses only the
final reviewed source. No production output was replaced.
Both paths were observed with the same timing wrappers around existing
functions; numerical kernels were not changed for instrumentation.

| Stage (seconds; frames 41 / 55 / 79) | Uncached | Cached, final |
| --- | --- | --- |
| Operator construction | 28.750 / 28.538 / 28.385 | 28.610 / 0 / 0 |
| Explicit cache compatibility check | none | 0.422 / 0.428 / 0.421 |
| All mesh checks combined | 0.844 / 0.852 / 0.849 | 1.278 / 1.272 / 1.265 |
| GLL/Jacobian derivatives and local curl | 25.195 / 24.734 / 25.136 | 25.002 / 24.753 / 24.812 |
| Scalar L2 projection | 0.128 / 0.124 / 0.122 | 0.125 / 0.128 / 0.124 |
| VTR writer | 0.117 / 0.101 / 0.082 | 0.113 / 0.102 / 0.084 |
| Complete frame | 57.108 / 56.264 / 56.423 | 57.090 / 28.307 / 28.143 |

The explicit check is part of the combined mesh-check row, which also includes
the unchanged checks in curl and visualization assembly; these rows must not
be added twice. The local-curl timer includes GLL differentiation, Jacobian
validation/transformation and curl arithmetic, not just matrix differentiation.
Complete frame time additionally includes native reads, visualization assembly,
source/output hashes, VTR/PVD read-back and publication. Batch wall time is
measured around the entire workflow call after imports and includes preflight.

Three-frame wall time decreases from **170.710 s to 114.455 s**, an actual
**1.492x speedup / 32.95% reduction**. Operator construction counts are 3 versus
1, confirmed by observers rather than inferred from metadata alone. The first
frame still pays initialization; the two subsequent frames save about 28 s
each. Process peak RSS is **1,205,374,976 bytes (1.123 GiB)** uncached versus
**1,193,365,504 bytes (1.111 GiB)** cached. These are independent process
high-water marks, not isolated per-stage allocations or a guaranteed memory
reduction. The retained operator budget is **187,291,640 bytes (178.6 MiB)**;
the computation preflight is **2,627,026,232 bytes**, with **225,280 bytes**
explicit mesh-validation scratch. No all-81-frame performance claim is made.

Each selected frame was read back and compared against both the uncached
calculation and the historical production volume. All 9,229,725 float64 scalar
values, min/max, coordinate vectors, dimensions `(1905,85,57)`, actual timestamps,
signed data and x-fast ordering agree exactly (`array_equal`, no tolerance).
Times remain 10.0035071079, 13.50377813401 and 19.50240408807. The cached files
also pass the full current metadata/source/signature/checksum/VTR/PVD validator.
All three coordinate vectors and dimensions match the existing full-volume
frame-79 lambda_ci grid, including first-cell point ordering and frame-79 time.
VTK reads in the comparison utility are preceded by the same RAM gate, with
already retained comparison arrays included in its budget.

The 19 new cache cases plus the additional exact RAM-fix predecessor case cover
single construction/lifetime, geometry/order/orientation/shape/count mismatch,
periodic and mapping mismatch, exact fresh/cached projection, all-reuse and
mixed resume, retained read-back RAM, late memory drops, and invalid cache RAM
arguments. All prior collision/overwrite, unknown-RAM and signature rejection
tests remain in place. Focused validation uses:

```bash
PYENV_VERSION=research312 python -B -m pytest -q \
  tests/test_scalar_projection.py tests/test_vorticity.py \
  tests/test_omega_x_workflow.py tests/test_omega_x_operator_cache.py \
  tests/test_l2_projection.py tests/test_lambda_ci_workflow.py -p no:cacheprovider
PYENV_VERSION=research312 python -B -m pytest -q -p no:cacheprovider
```

Final validation: **200 focused tests passed in 3.34 s**, and **1664 complete
suite tests passed in 30.19 s** (1644 existing plus 20 new cases). Tracked and
new-file whitespace checks pass. Historical normal resume on frames 41/55/79
completes with zero operator construction and no recomputation/export; all
nine full VTR reads across preflight, under-lock checking and reuse are gated
by RAM checks. Low/unknown RAM and a drop after the first cached frame fail
before the next full VTR/native read. All **686 existing files** (10 omega_x,
676 lambda_ci), including historical JSON metadata, retain their inventory and
SHA-256 values. All three raw snapshot hashes match their metadata. The five
scientific/export signature modules and `lambda_ci_series.py` remain
byte-identical to the starting commit.

The benchmark is a measured three-frame batch on this host, not a statistical
performance study. Moving meshes, reordered elements and changed configurations
remain unsupported for reuse and stop the run. Cache lifetime does not span
invocations, so resuming later starts a fresh cache only if computation is
needed. Allocator/page-cache behavior and competing processes may affect timing
and available RAM; resource checks do not reserve memory. Existing VTR byte
hashes remain strictly checked even though fresh serialization can contain VTK
object-address labels, as described above.

Atomic publication covers one frame, not an entire batch or sudden power loss.
Failed/interrupted staging is preserved for review; only completed published
frames are resumable. Backups and unknown files are never automatically deleted.

## Validate saved production and create a time-series PVD

`scripts/37_validate_omega_x_series.py` audits existing volumes and publishes
one ParaView Collection; it does not load native velocity payloads or calculate
omega_x. The reusable implementation is `nek_post.omega_x_series_pvd`.
Neither numerical kernels nor the existing workflow/signature guard is changed.

The production input root is
`/data/Nek5000_data/results/h_refinement/omega_x/case_N7_H/`.
The current 81-frame dataset has an additional case directory, so its frame
files actually reside at
`ROOT/case_N7_H/fNNNNN/case_N7_H_fNNNNN_omega_x.{vtr,pvd,json}`.
This layout is preserved. Recursive discovery accepts both direct and nested
frame directories, requires exactly one triplet for each index 1 through 81,
and rejects duplicate indices, unexpected scientific files and symlinks.
Operational locks and empty staging directories are not scientific outputs.
Every metadata output path must match the actual discovered triplet.

The audit establishes SHA-256 baselines of all existing VTR/PVD/JSON files
before full VTR reads and verifies them again after validation and before
publication. It reads all original `GC0.f00001` through `GC0.f00081` headers,
requires strictly increasing actual times, and verifies element count 26,112,
8 nodes per reference direction (P7), original source path/index/istep/hash,
periodic x/y configuration, completion metadata and exact implementation
signature compatibility. Missing required checksums are errors, not inferred
metadata additions. Source size/mtime are checked again after the audit.

Each frame passes the existing complete artifact validator and an additional
sequential VTK read-back for shared-grid checks and statistics. Expected
dimensions are `(1905,85,57)` with 9,229,725 float64 omega_x values and bounds
`[-17,17] x [0,1.5] x [0,1]`. Coordinates must be finite and strictly increasing,
match their recorded hashes and agree exactly across every frame. The VTR time
must equal the original header time. The implicit rectilinear x-fast convention
is checked using zero-based extents and point IDs/coordinates at the first,
adjacent and last x point of every y/z row. This checks serialization/indexing;
it does not independently recompute physical curl from the velocity field.

All scalars must be finite and match metadata min/max; both signs are not
required. Frame 1 legitimately contains zeros. Min/max/mean/RMS are recorded
per frame in the JSON report. Mean and RMS are arithmetic statistics over the
exported points, including retained periodic endpoints, not quadrature-weighted
volume statistics or enstrophy. Scaled, chunked float64 reductions avoid
unnecessary full-field temporary arrays and square overflow. Each VTK reader
is guarded by the existing host/cgroup RAM probe; unknown/insufficient RAM
fails closed. Only one volume and tiny reference coordinate vectors are held
at a time. The existing validator and statistics reader run sequentially.

Reproduce the read-only audit:

```bash
PYENV_VERSION=research312 python -B scripts/37_validate_omega_x_series.py \
  --case N7_H \
  --input-dir /data/Nek5000_data/results/h_refinement/omega_x/case_N7_H \
  --dry-run
```

Create the series after complete successful validation:

```bash
PYENV_VERSION=research312 python -B scripts/37_validate_omega_x_series.py \
  --case N7_H \
  --input-dir /data/Nek5000_data/results/h_refinement/omega_x/case_N7_H \
  --output-dir /data/Nek5000_data/results/h_refinement/omega_x/case_N7_H
```

`--case` also accepts an explicit source snapshot directory. `--input-dir`
defaults to the production case root above. `--output-dir` is the exact PVD
directory and defaults to the input root; a separate external directory is
allowed, while destinations inside frame outputs, source snapshots, this
repository or lambda_ci results are rejected. Dry-run creates no files, even
when the requested output directory does not exist. JSON diagnostics go to
stdout and progress to stderr; invalid datasets return exit status 1 with
missing/duplicate/invalid-frame errors. All 81 frames are required by the CLI;
the reusable test API permits explicit smaller index sets.

The target is `ROOT/case_N7_H_omega_x_series.pvd`. It contains 81 entries with
actual timestamps from **0.0 to 20.0**, in chronological order, using paths
relative to that PVD directory. For the current nested layout:

```xml
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
  <Collection>
    <DataSet timestep="0.0" group="" part="0"
      file="case_N7_H/f00001/case_N7_H_f00001_omega_x.vtr" />
    <!-- Frames 2 through 80 use their actual source times. -->
    <DataSet timestep="20.0" group="" part="0"
      file="case_N7_H/f00081/case_N7_H_f00081_omega_x.vtr" />
  </Collection>
</VTKFile>
```

Publication writes a unique temporary file in the target directory, flushes
and fsyncs it, parses its XML and verifies all reference paths/times, rechecks
the original artifact baseline, then uses atomic `os.link` no-replace creation
and directory fsync. An existing target, including a racing writer or symlink,
is never overwritten. Only this invocation's temporary file is removed.
There is no overwrite option. A failed audit never publishes a completed
series. A later dry-run can validate an already existing series at the input
root. External concurrent changes are detected by hashes/stats at audit
boundaries; this is not a filesystem snapshot or a substitute for keeping
production inputs stationary during validation.

The 2026-10-16 full-dataset dry-run validated all **81 VTR + 81 single-frame
PVD + 81 JSON** files, with no missing, duplicate, corrupted or unexpected
scientific frames. The frame-by-frame diagnostics and source/VTR/PVD SHA-256
values are recorded in [omega_x_N7_H_production_statistics.csv](/data/Nek5000_data/results/h_refinement/omega_x/case_N7_H/omega_x_N7_H_production_statistics.csv).
This link targets the external production dataset on this host, not a CSV
stored in the repository's `docs/` directory.
Across the actual times 0-20, the minimum of omega_x is
**-37.8430021494745** (frame 79), and the maximum is
**35.5031890168688** (frame 80). Pointwise mean ranges from
**-0.00073328310756719** (frame 34) to **0.0012319737378544086**
(frame 76). RMS ranges from **0** (frame 1) to **0.8180658030270787**
(frame 81). These statistics describe the exported sampling, not a weighted
volume average.

At the 2026-10-16 series-validation stage, focused workflow/cache/series tests
passed **93 cases with 1 optional skip**; the complete suite passed
**1693 tests with 1 optional skip in 29.77 s**. These are historical counts.
The 30 new series tests cover chronological actual times, missing/duplicate
frames, references, metadata/checksums/signatures, scalar dtype/finite values,
shared coordinates, signed/zero/one-sided fields, statistics, atomic failure
and publication races, collisions, read-only dry-run and artifact preservation.
The skipped pre-existing three-frame integration test looked for deleted
scratch outputs in the older direct `ROOT/fNNNNN` layout; the historical
real-data audit covered all 81 production frames at their actual nested paths.
The current synthetic replacement removes that external scratch dependency
without recreating the deleted frames or weakening the production validator.
The full repository suite now contains 1736 tests; current commands and results are
recorded in the linked P7 document.

The final publication completed successfully at
`/data/Nek5000_data/results/h_refinement/omega_x/case_N7_H/case_N7_H_omega_x_series.pvd`.
Read-back confirms exactly 81 unique relative references, increasing times
0-20 and valid ParaView Collection XML. All 81 single-frame PVD headers also
pass strict Collection-type checks. The new series SHA-256 is
`b1ab69ae98106a1b6a2abd55743687cd6c9fa2323afe3916ba8658655d4bf50d`.
In that historical publication audit, the pre-audit **921-file baseline**
(243 omega_x scientific files, two existing locks and 676 lambda_ci files)
remained byte-identical; the inventory added only the newly authorized series
PVD. The full publication audit and dry-run statistics agreed exactly with the
CSV. All 81 stored source checksums matched their original snapshots. No
original metadata or numerical implementation was modified. Tracked and all
four new files passed whitespace checks at that stage.
