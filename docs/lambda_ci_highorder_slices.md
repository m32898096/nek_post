# High-order lambda_ci slices (one frame)

Run from the existing research312 environment:

```bash
PYENV_VERSION=research312 python scripts/34_plot_lambda_ci_slices.py \
  --index 79 --y-midplane 0.75 --x-slice 15 --x-min 0 \
  --subdivisions-per-element 28 --vmin 0 --vmax 10 --save-samples
```

Default: 28 intervals / 29 points per element edge (four times the seven
native GLL8 intervals). `--samples-per-element N` retains the older point-count
interface and is mutually exclusive with `--subdivisions-per-element`.
`--no-paraview` retains the Python-only output naming. `--output-dir` sets the
final plots directory; its sibling `slices/` receives VTRs, and the parent of
the frame directory receives the case-level PVD collections. Output paths
inside the repository or the original case directory are rejected.

The existing numerical operators compute/project nine gradient components
once. Each component is evaluated using the native P7 GLL barycentric basis
on the physical plane, then eigenvalues produce lambda_ci and its square.
Coordinates use the original per-element profiles, including nonuniform
spacing and exact inverse mapping for the plane and any interior crop edge.
Shared points are merged after checking tensor agreement, without averaging.
Eigenvalues are chunked; no scalar smoothing or interpolation is used in
this reconstruction. Dense sampling preserves P7; it is not a new simulation.

VTR planes retain x/y/z physical coordinates, dimensions (Nx,1,Nz) or
(1,Ny,Nz), float64 PointData, x-fast indexing and TIME_VALUE. PNG/PDF and VTR
use exactly the same sampled scalar arrays. VTR cells themselves are linear:
dense sampling reduces the gaps between samples, without making the file a
native high-order element representation.

Open either case-level `*_series.pvd` in ParaView, click Apply and select
Surface with PointData `lambda_ci`. These datasets are already planes and
need no Slice filter. In the color-map preset chooser, Import the generated
`plots/*_tecplot_continuous.json`, select **NekPost Tecplot Continuous Blue
Red**, turn off discretization, and Rescale to Custom Range **0, 10**.
Use the same range/camera for comparisons. Do not add smoothing or Resample
To Image.

Outputs are exclusive: a repeated frame run fails before loading the snapshot.
The PVD append API preserves existing entries and rejects repeated timestamps
or file references. It supports later callers adding explicitly supplied
frames; this script still computes only one frame.

`sample_linear_vtr_reference` is a separate diagnostic helper for comparing an
old scalar VTR using physical trilinear interpolation at the new sample
coordinates. Differences are between representations, not exact-solution
errors; a scalar VTR cannot recover the projected gradient tensor.

Tests (synthetic data only):

```bash
PYENV_VERSION=research312 python -m pytest -q tests/test_lambda_ci_plotting.py tests/test_lambda_ci_workflow.py
PYENV_VERSION=research312 python -m pytest -q
```

## Sequential series and safe resume (Stage 5)

```bash
PYENV_VERSION=research312 python -B scripts/35_compute_lambda_ci_slice_series.py \
  --start-index 1 --end-index 81 --resume --dry-run
PYENV_VERSION=research312 python -B scripts/35_compute_lambda_ci_slice_series.py \
  --indices 65 --resume --log-name stage5_smoke65
PYENV_VERSION=research312 python -B scripts/35_compute_lambda_ci_slice_series.py \
  --indices 17,33,49 --resume --log-name stage5_small_batch
PYENV_VERSION=research312 python -B scripts/35_compute_lambda_ci_slice_series.py \
  --start-index 1 --end-index 81 --resume --log-name stage5_production
```

`--output-root` is the **case-level** directory. Defaults retain the confirmed
planes, 28 subdivisions, periodic x/y projection and color range [0,10].
`--max-frames` limits a selection, and `--save-samples` retains optional NPZs.
The single-frame script remains available.

A process builds its L2 projection operator once and retains quadrature weights,
lumped masses and element-order-specific mapping. The two interpolation plans
are built once during preflight and reused. Original coordinate signatures are
checked before every new frame's gradients; any changed geometry or element
ordering stops publication. The established element gradient API remains
unchanged. Only the two planes undergo eigenvalue/scalar evaluation; no full
volume lambda_ci or 3D visualization is produced.

`--resume` validates header/source checksums, completed diagnostics, numerical
implementation signatures (new results), exact sample coordinates, VTR point
arrays and squared values, PNG decoding, PDF page decoding and continuous RGB
controls. Valid legacy frame-79 metadata is accepted after equivalent checks.
Existing incomplete/incompatible frame directories stop execution and remain
untouched; preserve/move them explicitly after review before retrying.

New results are written under `.stage5-staging/` and decoded/checked before
Linux atomic directory publication with no replacement. A case-level advisory
lock prevents simultaneous series writers. A crash before publication leaves
staging intact. `--recover-staging` can publish a completed valid staged frame;
incomplete staging is preserved and a new unique staging directory is used.
A crash after publication but before PVD/checkpoint update is repaired by the
next validated `--resume`. Final verification identifies each collection by its
explicit `y_midplane` or `x_slice` series key, never by a filename substring;
nondefault planes such as `--y-midplane 0.6` retain the same frame/time checks.
Existing PVD entries are retained, duplicates rejected and entries sorted by
actual physical time. Output filenames are unchanged.

PVD updates use a uniquely named temporary file beside the destination. The
complete XML is flushed, fsynced, parsed, and checked against the intended
entries before atomic replacement. A new collection is published by an atomic
no-replace hard link. A failure before publication leaves the previous PVD
intact (or leaves a new destination absent); only the current operation's
temporary is cleaned up. Legacy `.pvd.append` files and unknown orphan
temporaries are ignored and preserved, so they do not block resume.

Each helper update holds a persistent `.<pvd-name>.lock` advisory lock; the
case-level series lock remains in place. All writers must use this helper or
honor its lock; unique temporary names alone do not prevent lost updates.
Lock sidecars must not be removed while writers may be active. Publication is
atomic per PVD, not across both collections or the checkpoint: validated resume
repairs missing entries after a process interruption. An already malformed
destination PVD still stops validation and requires explicit review; this is
not a power-loss durability guarantee. Operational JSON checkpoints continue
to use atomic replacement.

Logs: `logs/<log-name>.jsonl`, `logs/<log-name>_summary.json` and the latest
`processing_summary.json`. Summaries contain per-frame actual time, both scalar
min/max, runtime, generated/reused/recovered status, observed RSS, peak RSS and
operator/plan build counts. Final verification uses the expected manifest roles
rather than counting unrelated files by extension, and rechecks all source and
pre-existing result checksums. A run is complete only when summary status is
`completed` and final validation is `passed`.

For a detached launch when tmux is unavailable, redirect stdout/stderr to a log
outside the repository and start a new session with stdin disconnected. Inspect
both the process PID and its checkpoint; a live PID alone does not imply success.
