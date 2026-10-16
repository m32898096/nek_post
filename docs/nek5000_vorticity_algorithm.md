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
R = 0 for reused frames
R = bytes of the retained native field and coordinates for staging read-back
M_frame = M_validation                             (reused frame)
M_frame = max(M_computation, M_validation)          (new/overwrite frame)
M_batch = max(M_frame over requested frames)
```

The four-payload allowance covers file/decode buffers, VTK dataset arrays,
NumPy conversion and validation scratch. `vtk_to_numpy` often returns views;
the conservative estimate does not require zero-copy behavior. The existing
1-GiB margin covers bounded checksum buffers, library overhead and uncertainty.
The unchanged computation estimate includes reader/source buffers, local raw
and projected scalars, local mass weights, assembled mass/RHS, visualization
storage and its original safety margin. The batch takes a maximum because
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
The RAM fix permits one explicitly reviewed omega_x workflow transition:
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
altered synthetic files even after their fixture checksum is refreshed. An
optional read-only integration test validates the actual historical frames
41/55/79 when those inputs and outputs are available; it ran and passed in
this checkout and verifies RAM checks before all three reader constructions.

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

Projection operators are still rebuilt per new frame (previously measured at
about 28-29 seconds); persistent operator caching is outside this RAM fix.

Atomic publication covers one frame, not an entire batch or sudden power loss.
Failed/interrupted staging is preserved for review; only completed published
frames are resumable. Backups and unknown files are never automatically deleted.
