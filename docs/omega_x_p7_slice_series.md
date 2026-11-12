# Saved Omega_x P7 midplane slice series

## Scope and numerical method

`scripts/39_compute_omega_x_p7_slice_series.py` extends the validated
[single-frame workflow](omega_x_p7_midplane_slice.md). It reads existing
mass-lumped-L2-projected `omega_x` from native GLL VTRs, processes frames
sequentially and uses the same `reconstruct_p7_midplane` numerical evaluation.
It never loads native velocity payloads or computes derivatives, curl, L2
projection or eigenvalues. Original snapshots are read only for header and
streaming SHA-256 provenance checks.

The CLI uses N7_H, frames 1-81, y=0.75, strict `0 < x <= 17`, z=0-1, P7 and
29 samples per SEM edge including endpoints. Each slice has dimensions
`(3808,1,225)`. It retains physical float64 coordinates, x-fast ordering,
signed float64 `omega_x`, `omega_x_linear`, `omega_x_p7_minus_linear` and the
actual simulation timestamp in `TIME_VALUE`.

One invocation-local `P7SlicePlan` retains the validated eight-node/stride-7
SEM profiles, plane bases and existing x/z interpolation matrices. Every
frame must match the cached coordinate vectors exactly before use. A changed
topology, ordering, coordinate profile or configuration stops the run; the
workflow does not silently rebuild the plan. The arithmetic in the scalar
evaluation loop is unchanged. Shared traces are checked, without averaging,
filtering, smoothing or overshoot clipping.

For optional volume reconstruction, see the
[single-frame 3D VTU exporter](vortex_3d_binary_vtu.md#phase-2-p7-volume-reconstruction).
It interpolates the same projected-curl P7 scalar on denser GLL nodal sets and
audits agreement against the curl of reconstructed projected gradients. This
does not change the saved-slice workflow or its uniform 29-point sampling.

## Input and output layout

The default native input root is:

```text
/data/Nek5000_data/results/h_refinement/omega_x/case_N7_H/
  case_N7_H/f00001/case_N7_H_f00001_omega_x.{vtr,pvd,json}
  ...
  case_N7_H/f00081/case_N7_H_f00081_omega_x.{vtr,pvd,json}
```

Discovery also supports a direct `ROOT/f00001/` layout, but rejects missing
frames, duplicate indices, symlinks, incorrect frame numbering and incomplete
native triplets. The full selection requires exactly source snapshots 1-81.
Header times must agree with native metadata, VTR and PVD times and increase
strictly with frame numbering; neither filename order nor an assumed time
increment substitutes for actual time.

The default output root is:

```text
/data/Nek5000_data/results/h_refinement/omega_x_highorder_slices/case_N7_H/
  f00001/case_N7_H_f00001_omega_x_y0p75_xgt0_n29_*
  ...
  f00079/  # Existing single-frame artifacts are preserved.
  ...
  f00081/
  case_N7_H_omega_x_p7_slice_series.pvd
  .omega-x-p7-staging/
  logs/<invocation-id>.json
```

Each completed frame retains the existing eight-file contract: one three-field
VTR, validation JSON, and PNG/PDF pairs for P7, native linear and comparison
views. Rendering uses the existing per-frame symmetric color range shared by
the P7 and linear representations; the range can change between frames.
No global fixed-color-range animation claim is made.

`--input-dir` is the exact native case root. `--output-dir` is an external
parent root to which `case_N7_H` is appended, matching script 38. Output roots
inside native production, original snapshots, this repository or lambda_ci
results are rejected. The resolved output root and every actual ancestor are
also checked for existing per-frame omega_x/lambda_ci artifacts. An output
equal to or below a scientific frame is rejected before creating directories,
including through symlinks, renamed frame directories and nonexistent nested
paths. A case-level root containing frame subdirectories remains valid, as do
independent sibling paths; checks use resolved ancestry, not string prefixes.
There is no overwrite option.

## Resume, signatures and recovery

Without `--resume`, any requested existing frame stops preflight. With resume,
the complete native/source provenance is checked before accepting completed
slices. Validation requires source snapshot and native artifact SHA-256 values,
producer signatures, output checksums, exact physical grid, time, scalar dtype,
finite values, active scalar, exact difference array, extrema diagnostics,
decodable images/PDFs and the complete output manifest.

New metadata uses schema version 2 and records the full-file SHA-256 values of
the interpolation, geometry, GLL and orchestration dependencies. Any mismatch
rejects reuse; signatures are not disabled to accommodate code edits.
Dry-run reports and all invocation checkpoints also carry the same top-level
`interpolation_signature` from the existing six-file signature mechanism.
Signature changes during preflight, lock waiting or execution stop the run.

The historical frame-79 JSON has no interpolation-code signature. Its explicit
compatibility assessment is narrowly bound to case/frame and these exact
published metadata bytes:

```text
34d0d560cac1930a676892b88efe040e1768d0472ffcf9e8122c272441c07d41
```

The single-frame module at the start of this extension had SHA-256
`552a4861e9d3a24b9aede22d646e3d77519923f41f677657caafcad5cd4555a4`.
The extension extracts its coordinate preparation into a plan and its native
validation/read into a shared helper. It retains the interpolation arithmetic.
On every invocation that reuses the historical report, all original provenance
and artifact checks must pass, and reconstructed P7/linear arrays must match
the saved arrays exactly. Unknown legacy reports are rejected. This assessment
is invocation-local and never rewrites historical metadata or adds a relaxed
global producer-signature exception.

New frames are written to unique private staging directories. All eight files
are validated before Linux atomic no-replace directory publication. Failed
staging is preserved. `--resume --recover-staging` validates a uniquely complete
staged frame against current sources, configuration and signatures before
publication. Incomplete staging is preserved and a fresh directory is used;
multiple complete candidates or incompatible complete staging require review.

Resume reconstructs progress from validated frame directories; logs are
diagnostics, not trusted recovery manifests. Every invocation has its own JSON
checkpoint. A persistent advisory lock serializes cooperating series writers.
Do not delete a lock inode while a writer may be active.

Initial preflight remains read-only. After obtaining that lock, execution runs
the full preflight again and uses only its refreshed frame actions, staging
classification, provenance and RAM/disk estimates. Frames completed by another
writer while waiting are validated and reused without reconstruction. If all
frames became complete, no new scalar reconstruction is performed and the
pending-output estimate is zero. An approved historical report still needs its
exact-array assessment once per invocation; the result is retained across the
lock refresh while its metadata, source and output checksums are revalidated.

Fixed sources and previously completed outputs must remain unchanged while
waiting. A complete staged frame may move to its final location under the
other writer's lock; the fresh preflight validates that location rather than
requiring stale staging paths to exist. Newly incomplete/incompatible final
frames fail closed. Complete staging is recovered only with
`--resume --recover-staging`; incomplete staging remains untouched and a new
private staging directory is used. Lock-time validation failures occur before
an execution checkpoint is created, though the persistent lock may exist.

## Complete PVD publication

Only a validated selection containing exactly indices 1-81 can publish
`case_N7_H_omega_x_p7_slice_series.pvd`. Subset smoke runs report
`completed_subset`, with zero PVD entries and no partial production collection.
All target frames and protected inputs are checked again before publication.

The PVD contains 81 unique relative references such as
`f00079/case_N7_H_f00079_omega_x_y0p75_xgt0_n29_p7.vtr` with actual, strictly
increasing timestamps. A unique same-directory temporary XML file is flushed,
fsynced and parsed before atomic no-replace publication. Existing valid PVDs
are validated and reused; malformed files, symlinks and publication collisions
are preserved and rejected. A failure before publication leaves no completed
series PVD; already completed frame directories remain resumable.

In ParaView, open this PVD, click Apply, select Surface and Point Data
`omega_x`, and view along the y axis. The PVD supplies animation times directly;
no Slice filter or high-order 3D volume is needed. Use a consistent custom
signed color range for cross-frame visual comparisons.

## Resources and commands

Dry-run performs real checksum and VTR validation sequentially, including any
existing frame's compatibility assessment. It creates no output directory,
lock, checkpoint, staging or PVD. Streaming original snapshot hashes can read
substantial data; it does not decode velocity payloads.

RAM checks precede full VTR reads. The budget retains one native frame and
one plan, not all 81 scalar arrays. It uses the existing four-payload VTR
validation allowance and 1-GiB margin plus conservative reconstruction/figure
scratch. Disk preflight reserves conservative three-field VTR and figure space
for pending frames plus a 1-GiB margin, and rechecks remaining space during
execution. Availability checks do not reserve RAM or disk against other jobs.

Read-only validation of the canonical 81-frame input and existing frame 79:

```bash
PYTHONDONTWRITEBYTECODE=1 PYENV_VERSION=research312 \
  python -B scripts/39_compute_omega_x_p7_slice_series.py --resume --dry-run
```

Isolated smoke reconstruction (choose a new external destination):

```bash
PYTHONDONTWRITEBYTECODE=1 PYENV_VERSION=research312 \
  python -B scripts/39_compute_omega_x_p7_slice_series.py \
  --indices 1,41,79 --output-dir /tmp/omega-x-p7-smoke-new
```

Full production command, for a separately approved run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYENV_VERSION=research312 \
  python -B scripts/39_compute_omega_x_p7_slice_series.py \
  --resume --recover-staging
```

Tests use deterministic small P7 snapshots and saved scalar VTRs in pytest
temporary directories. They cover actual times, numbering, missing/duplicate
frames, signature/checksum rejection, geometry changes, exact cached/cold
interpolation, signs/zero-compatible validation, read-back, recovery, memory
and disk protection, no curl recomputation, and the 81-entry final-publication
gate. Production reconstruction is not run by pytest.

## Initial validation before orchestration fixes (2026-11-05)

- Focused P7/series/workflow tests: **129 passed in 14.69 s**, including 42 new
  series cases; zero failures or skips.
- Full repository suite: **1778 passed in 45.85 s**; zero failures or skips.
- Canonical read-only preflight: **81 frames validated**, actual times 0-20,
  one prepared plan, **80 planned reconstructions and one compatible reuse**
  (frame 79). No canonical output was generated. The validation took 199.92 s,
  including full source hashing and sequential native-volume checks.
- Resource estimate: **2,246,642,524 bytes RAM**; **4,632,721,280 bytes** of
  conservative pending output space plus the **1,073,741,824-byte** disk margin.
- Isolated real-data smoke: frames **1, 41 and 79**, actual times **0.0,
  10.0035071079 and 19.50240408807**, all `(3808,1,225)`, completed in 20.27 s
  including preflight. Per-frame processing took 4.86, 3.64 and 3.59 s on this
  host, with one plan build. These three-frame measurements are not an
  all-81-frame production benchmark or a RAM high-water measurement.
- The smoke outputs' three scalar arrays and physical coordinates match the
  committed pre-extension single-frame algorithm **exactly** (`array_equal`,
  maximum absolute difference zero). The frame-79 smoke arrays also match the
  preserved canonical frame-79 arrays exactly. Frame 1 remains identically zero;
  frames 41 and 79 retain both signs. VTR read-back and recorded output hashes
  passed. Subset execution published no PVD.
- A before/after SHA-256 inventory of **255 existing native/high-order files**
  showed no changes, additions or removals; size/mtime records for all 81 raw
  snapshots were unchanged. The canonical high-order tree still contains only
  `f00079`, with no production P7 series PVD.

Smoke artifacts and detailed review reports were retained under
`/tmp/omega-x-p7-series-review-9jr7tqgf/` for this session. That is temporary
evidence, not a required input or a durable repository artifact; the numerical
results above and self-contained tests do not depend on retaining that path.
The full 81-frame production reconstruction was not launched.
These initial temporary smoke metadata signatures identify the implementation
before the orchestration fixes; current code rejects them for resume. Fresh
post-fix evidence must use outputs generated by the current implementation.

## Validation after orchestration fixes (2026-11-05)

- Added 18 regression cases covering resolved frame ancestry, exact-root and
  renamed-frame protection, independent/canonical case roots, seven synchronized
  writer scenarios, invocation signatures/checkpoints and lock-time resource
  depletion. The initial regressions reproduced the issues before source edits.
  Running the final regression set against the original staged source gives
  **17 failures and one passing acceptance control**. All 18 pass with the fix.
- Concurrency tests use events and the actual advisory writer lock, with no
  timing sleeps. They cover partial/all completion, incompatible/incomplete
  final output, incomplete/complete staging and staging published while waiting.
  Reconstruction counters verify reuse, recovery and zero work when all frames
  become reusable; checksums verify the first writer's scientific files survive.
- Focused series/single-frame/workflow tests: **147 passed in 17.27 s**.
  Full repository suite: **1796 passed in 48.86 s**, zero failures or skips.
- Fresh canonical read-only preflight: **81 validated**, actual times **0-20**,
  **80 pending and frame 79 reused**, one plan, **175.86 s**. The RAM estimate
  remains **2,246,642,524 bytes** and pending disk estimate **4,632,721,280 bytes**
  plus the **1,073,741,824-byte** margin. No production reconstruction or PVD
  publication was performed.
- Fresh isolated smoke frames **1, 41, 79** completed in **15.46 s**, with one
  plan, no subset PVD, and exact equality of all three arrays, coordinates and
  timestamps against the committed pre-extension single-frame implementation.
  Maximum scalar differences were zero. Frame 79 also matched canonical arrays
  exactly. Modern smoke resume validated all three without reconstruction.
- Current reports, checkpoints and smoke metadata share the existing six-file
  signature; the series module SHA-256 is
  `17228defc3aa972c9ee5934c7319ce15305eacd30f3f8bfebd5a143086e14d6a`.
  Prior temporary smoke outputs fail current signature validation and remain
  untouched. The pinned historical frame-79 report is unchanged.
- A fresh before/after checksum inventory preserved **all 255 existing
  native/high-order files** with no additions or removals. All 81 source
  size/mtime records are unchanged. The canonical case root still contains only
  the original eight-file `f00079` directory, with no P7 series PVD or new lock,
  log or staging directory. The numerical module and dependencies are unchanged
  from the initial staged snapshot for these orchestration fixes.

Detailed temporary evidence and fresh smoke outputs are under
`/tmp/omega-x-p7-series-fix-review-LuE8S7/`. These files are session evidence,
not durable workflow inputs. Full production remains a separate operation.

## Numerical and operational limits

P7 interpolation represents the existing projected field; it does not increase
DNS resolution or establish improved physical accuracy. Polynomial overshoot,
C0 interfaces, native oscillations and finite-sample rasterization can remain.
The global native importer still requires the validated affine axis-aligned P7
structured topology. Rectilinear VTRs do not recover original Nek storage order.

Publication is atomic per frame and per PVD, not across the entire batch, and
is not a power-loss durability guarantee. Checksums/stat checks detect changes
at validation boundaries, not through a filesystem snapshot. Cooperating
writers must honor the lock. Existing malformed outputs are never repaired
automatically. Formal ParaView GUI behavior remains subject to the previously
documented application-level limitations.
