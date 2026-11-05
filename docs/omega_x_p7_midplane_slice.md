# Saved omega_x: P7 midplane reconstruction and interpolation comparison

## Scope and provenance

This workflow reads the **already mass-lumped-L2-projected native scalar** from
`/data/Nek5000_data/results/h_refinement/omega_x/case_N7_H/`.
For the validation on 2026-10-16, the actual input is
`case_N7_H/f00079/case_N7_H_f00079_omega_x.vtr` beneath that root, together with
its existing JSON and single-frame PVD. The source header and checksum are
checked against `/data/Nek5000_data/case_N7_H/GC0.f00079` without loading its
velocity payload. Frame 79 has actual simulation time **19.50240408807**.

No velocity gradients, Jacobian derivatives, curl, L2 projection, or eigenvalues
are recomputed. No existing scientific module or signature allowlist is edited.
The current omega_x workflow SHA remains
`8b7b441f7eb45d7deeb3159b98d256502231b0377151ae1fa7214997241d6176`.
Existing numerical signatures, source checksum, VTR/PVD artifact checksums,
coordinate hashes, finite scalar checks, extrema, and timestamps are validated
by the unchanged `validate_omega_x_frame` before reconstruction.

Reproduce the single-frame reconstruction and validation with:

```bash
PYENV_VERSION=research312 python -B scripts/38_plot_omega_x_p7_midplane.py \
  --case N7_H --frame 79
```

This command writes new artifacts; it is not a read-only inspection command.
The validated default frame-79 destination already exists, so the command will
refuse to overwrite it. To reproduce the run, supply `--output-dir` with a new
external root; case and frame directories are appended. Inspect the existing
VTR and validation JSON for a read-only review.

The default destination is
`/data/Nek5000_data/results/h_refinement/omega_x_highorder_slices/case_N7_H/f00079/`.
An existing destination is refused, including symlinks. `--output-dir` selects
an alternative root; case and frame directories are appended. Outputs inside
the source tree, production omega_x tree, lambda_ci tree, or repository are
forbidden. New files are created exclusively in a private staging directory,
validated, and atomically published without replacing a destination. Failed
staging is retained; the workflow does not clean scientific output directories.
RAM is checked before every complete VTR read, including output read-back;
unknown or insufficient RAM fails closed. The frame-79 preflight budget was
2,246,642,524 bytes, including the existing 1-GiB safety margin.

## Native mesh, SEM grouping, and physical plane

The native VTR is a canonical physical x/y/z rectilinear grid with dimensions
**1905 x 85 x 57** and a signed float64 PointData scalar `omega_x`. Its flattened
point order is x-fast; the NumPy field shape is `(Nz, Ny, Nx)`.
The validated element counts are `(272, 12, 8)`, with eight GLL nodes per axis
inside each P7 spectral element. Each global coordinate length must equal
`7 * element_count + 1`. Original SEM intervals are recovered at **stride 7**,
not from individual VTK cells. Each eight-node physical profile must be finite,
strictly increasing, and match an affine GLL profile within the established
storage-aware geometry tolerance. Reversed or incompatible geometry is rejected.
The original element storage order cannot be recovered from a rectilinear VTR;
this workflow uses its validated canonical physical ordering and producer
provenance, rather than inferring a raw Nek element ordering.

Native coordinate residuals against affine GLL profiles were
`(6.0419507619e-7, 3.6325838559e-8, 3.8309266537e-8)` in x/y/z. These reflect
the stored coordinate precision and satisfy the existing geometry tolerance.
Dense physical coordinates are evaluated from the original coordinate profiles;
they are not replaced by idealized affine coordinates.

The requested plane is `y=0.75`, `0 < x <= 17`, `0 <= z <= 1`.
For this dataset, `y=0.75` is **exact native y index 42** and the common boundary
of zero-based y elements **5 and 6** (one-based layers 6 and 7). Both traces are
evaluated and checked for agreement. No nearest-plane substitution is used.
For an interior y plane, the existing one-dimensional polynomial
coordinate inversion (`_physical_basis`, using a bracketed root solve) recovers
the reference coordinate before element-local P7 interpolation; extrapolation
is rejected.

## Interpolation and controlled comparison

The reconstruction reuses GLL nodes, barycentric weights/bases, physical-plane
inversion, and dense axis/element indexing from the established lambda_ci slice
architecture. Each scalar element is first evaluated in y, then by the
tensor-product x/z P7 matrices. Shared endpoints are stored once. Both sides
of each shared trace must agree within a scale-aware floating-point tolerance;
values are neither averaged nor smoothed at interfaces.

Each in-plane element edge is sampled at **29 uniformly spaced reference
coordinates including endpoints**, giving 28 intervals. This does not increase
the polynomial degree beyond seven. There are 136 positive-x elements and
eight z elements. The x=0 sample is removed strictly: the final x vector has
3,808 points, from `0.004464286157859132` through 17; the z vector has 225
points, including 0 and 1. There are no duplicate interface coordinates or
missing/NaN samples.

The comparison interpolates the original native rectilinear scalar with
`RegularGridInterpolator(method='linear', bounds_error=True)` at **exactly the
same final physical `(x, y, z)` points**. At the actual native y plane this
reduces to bilinear x/z interpolation; the implementation also supports
trilinear interpolation at interior y. Both images share coordinates, frame,
time, orientation, equal physical aspect, and a symmetric signed `RdBu_r`
scale, here **[-10.336129496368477, +10.336129496368477]**. Sample cells are
rendered with nearest shading without antialiasing, filtering, or smoothing.

The difference from lambda_ci is scientific: lambda_ci interpolates its nine
projected gradient components before computing eigenvalues. Here omega_x is
an already projected scalar, so its scalar P7 polynomial is evaluated directly.
No lambda_ci scalar interpolation or eigenvalue computation is introduced.

## Historical frame-79 validation results (2026-10-16)

The published `_validation.json` retains the interpolation diagnostics below.
The independent dense-to-native round-trip and ParaView probes were additional
historical checks, separate from pytest and from the persisted JSON; their
temporary reports are no longer available, as explained below.

| Quantity | Result |
|---|---:|
| Maximum absolute P7-minus-linear difference | 1.6814586860522178 |
| RMS difference over identical final samples | 0.05099450723780673 |
| Mean difference | 0.000370249583774337 |
| P7 RMS | 0.5734147684338333 |
| Linear minimum / maximum | -10.14046718939488 / 8.952608591718258 |
| P7 minimum / maximum | -10.336129496368477 / 9.224057952007572 |
| Maximum shared-element trace difference | 1.7763568394002505e-15 |
| Plane coordinate residual | 0 |
| Native nodal basis reproduction error | 0 |
| Independent dense-to-native round-trip maximum error | 1.1046719095020308e-14 |
| Independent dense-to-native round-trip RMS error | 2.802370670163045e-16 |
| Maximum local undershoot / overshoot | 0.43856404020115125 / 0.3455295755523571 |

RMS is an unweighted mean over these target samples, not a volume integral or
an error against an exact DNS solution. The RMS difference is about 8.89% of
the P7 sample RMS. Overshoot is measured against the local eight-by-eight
native scalar trace after y evaluation, not against the whole-volume extrema.
There were 1,916 overshooting trace-patches; because the requested y plane
lies on a shared boundary, the two y traces count the same physical x/z patch
twice. **No overshoot is clipped.**

The additional historical independent real-data round-trip checked all 1,088
positive-x SEM patches, recovering the P7 polynomial from eight dense targets per axis
and evaluating it back at all original GLL nodes (69,632 element-local values,
including shared nodes). The first x patch used eight strictly positive targets
because x=0 is not stored. Its nonzero round-trip error above is independent
of the exact native-node identity property in the main metadata.

The linear minimum occurs at `(15.75892890258399, 0.17574285460910874)` and
maximum at `(10.625, 0.18735714655451113)`. The P7 minimum occurs at
`(15.75892890258399, 0.1796142856235616)` and maximum at
`(10.633928902583989, 0.183485716662484)`. Large coherent structures occupy
the same regions. Full-domain and local inspections near x=10.6 and x=15.7
show changed contour corners and extrema under P7, while some element-scale
bands and finite-sampling pixelation remain. This supports an interpolation
contribution to the appearance, but does **not** establish that all sawtooth
features originate in ParaView, or that P7 is more physically accurate.

## Output contract and ParaView

All eight files share the deterministic stem
`case_N7_H_f00079_omega_x_y0p75_xgt0_n29`:

- `_p7.vtr`: true 2D rectilinear dimensions **(3808, 1, 225)**, 856,800 points,
  y coordinate exactly 0.75; signed float64 active PointData `omega_x`.
  Additional float64 arrays are `omega_x_linear` and `omega_x_p7_minus_linear`.
- `_p7.png`, `_p7.pdf`: P7 representation.
- `_linear.png`, `_linear.pdf`: native piecewise-linear representation.
- `_comparison.png`, `_comparison.pdf`: side-by-side, identical color limits.
- `_validation.json`: source hashes/signatures, physical coordinates/diagnostics,
  RAM budget, rendering settings, output paths/checksums, and read-back status.

The VTR is directly openable in ParaView, with x-fast ordering and the existing
project convention `FieldData/TIME_VALUE = 19.50240408807`. Full VTK read-back
checked exact scalar arrays, dtype, active scalar, coordinate vectors, dimensions,
point ordering, and timestamp. PNG decoding and Poppler PDF decoding succeeded;
all three final PDFs were rendered and visually inspected.

To display the existing plane in ParaView:

1. Open
   `/data/Nek5000_data/results/h_refinement/omega_x_highorder_slices/case_N7_H/f00079/case_N7_H_f00079_omega_x_y0p75_xgt0_n29_p7.vtr`
   with the XML Rectilinear Grid reader and click **Apply**.
2. Select **Surface** and color by **Point Data / omega_x**. View along the y
   axis and reset the camera to see the x-z plane. No Slice filter is needed.
3. For the documented signed comparison, use **RdBu_r** and the common range
   **[-10.336129496368477, +10.336129496368477]**. Select `omega_x_linear` in
   the same file to inspect the native linear representation at the identical
   physical samples; `omega_x_p7_minus_linear` stores their signed difference.
4. Do not add smoothing or filtering. For time metadata, use **Time Array =
   TIME_VALUE** where the reader exposes that property.

Historically, **ParaView 6.1.0-RC1** was additionally checked using
`pvpython --no-mpi --disable-registry`. Its direct XML reader returned exactly
the same primary scalar byte SHA-256 as the Python VTK reader:
`b66a86fce6153b6312ec8671ae75edf1c6bb4733c8d03e97576e1976c7b3df86`.
The direct reader and Extract Surface pipeline had correct physical bounds,
signed values, 856,800 points and 852,768 surface cells. Extract Surface uses
float32 display coordinates by default; the original VTR remains float64.
For animation time, select the reader's **Time Array = TIME_VALUE** and update
pipeline information; its default `TimeValue` name does not discover this
project's existing uppercase field. No temporal PVD is needed for this one plane.

A separate RC1 `servermanager.Fetch` probe of the degenerate-middle-axis
rectilinear grid returned inconsistent coordinates/scalars, despite correct
direct-reader and surface-pipeline data. Do not use that transfer path as a
validation oracle for this file. Final ParaView 6.1.0 GUI rendering was not
performed in that validation, and the 2026-11-05 review does not repeat a GUI
check; the historically tested build is RC1. This is an application-level
limitation, not a reason to change the required 2D geometry into a volume.

The input frame-79 VTR SHA-256 is
`47895d213ad75335cbc619e659093322786541073e7f7316749d1a352dbc351e`;
the new P7 VTR SHA-256 is
`74614930377b2b31d5571b78ec40975696af30e57b9483c4ff9826f6ccc249d0`.
In the 2026-10-16 validation, before/after complete-file SHA-256 inventories of
both original omega_x and lambda_ci case trees contained **923 files**, with zero changes, additions,
or removals. Source snapshot SHA also remained unchanged. Only real frame 79
was processed; no deleted scratch frames or 81-frame series were regenerated.
Supplemental session evidence was stored under
`/tmp/omega-x-p7-midplane-review/`, including
`frame79_report.json`, `original_artifacts.json`, `original_artifacts_after.json`,
`preservation_report.json`, `nodal_roundtrip.json`, `paraview_readback.json`, and
rendered/detail inspection PNGs. That temporary directory is no longer present
as of 2026-11-05. Its reports, including the independent round-trip and ParaView
probe, are historical evidence rather than currently accessible artifacts.
The published VTR, images/PDFs and `_validation.json` remain the durable output
record; pytest does not depend on the deleted temporary directory. A current
read-only check of those files does not repeat the historical 923-file audit.

### Read-only pre-commit review (2026-11-05)

The existing native and P7 VTRs were read without regenerating outputs. Source
header/time, current producer signatures, physical coordinates, signed float64
PointData, dimensions, x-fast ordering and the exact stored difference array
passed checks. All 11 recorded checksums matched: the native VTR/PVD/JSON,
original snapshot and seven non-metadata reconstruction outputs.

Re-evaluating the native linear field at all 856,800 exported physical samples
matched `omega_x_linear` exactly. An independent Legendre-basis evaluation at
2,176 points, two per positive-x SEM patch, differed from saved P7 values by at
most `5.329070518200751e-15`; its physical-coordinate residual was at most
`7.105427357601002e-15`. This check did not call the production barycentric
reconstruction, compute derivatives/projection, or write output files.
Recomputed max-absolute and RMS P7-minus-linear differences matched the table
above. Mean difference and P7 RMS differed from the JSON only by floating-point
reduction roundoff (`5.42e-20` and `1.11e-16`). This review did not repeat the
historical full-tree preservation audit or ParaView GUI validation.

## Pytest skip investigation and resolution

Before the synthetic replacement, the historical complete suite
(`python -B -m pytest -q -rs -p no:cacheprovider`,
with `PYENV_VERSION=research312`) reported **1693 passed, 1 skipped**.
The only skipped test was
`tests/test_omega_x_workflow.py::test_real_predecessor_three_frames_readonly_resume`,
with reason `Optional existing N7_H frame 41/55/79 integration data is unavailable`.
It depended on the deleted scratch frames at hardcoded paths and on an old
workflow SHA, while the retained production data live in the nested case layout.

`test_synthetic_predecessor_three_frames_readonly_resume` now builds deterministic
two-element P7 inputs and three small frames **only inside pytest's tmp_path**.
It assigns the explicitly approved predecessor SHA only to this generated test
metadata, then preserves all original assertions: three read-only resumes,
actual-time ordering, forbidden native payload/curl loads, low-RAM rejection
before any VTR reader, required RAM checks before each normal VTR read, and
unchanged checksums of all nine synthetic artifacts. These are small unit-test
fixtures, not re-created N7_H scratch data. Scientific production metadata and
the exact signature guard, including rejection tests for unknown SHAs and
changed numerical modules, remain untouched.

The 42 new focused cases cover constants; tensor polynomials through degree
seven in every coordinate; boundary and interior-y evaluation; independent
native GLL nodal reproduction; physical coordinate/orientation/grouping checks;
strict positive-x cropping; trace continuity; reported/unclipped overshoot;
signed 2D VTR arrays/order/time; no overwrite; RAM rejection before read-back;
missing, duplicate, incompatible and protected/symlinked input/output paths;
source preservation on rejection; CLI errors; and controlled PNG/PDF decoding.

```bash
PYTHONDONTWRITEBYTECODE=1 PYENV_VERSION=research312 python -B -m pytest -q -rs \
  tests/test_omega_x_p7_slice.py tests/test_omega_x_workflow.py -p no:cacheprovider
# 2026-11-05: 87 passed in 3.26s; zero failures and zero skips.

PYTHONDONTWRITEBYTECODE=1 PYENV_VERSION=research312 \
  python -B -m pytest -q -rs -p no:cacheprovider
# 2026-11-05: 1736 passed in 29.96s; zero failures and zero skips.
```

Optional real-data validation is the explicit CLI command above, separate from
self-contained pytest. It requires the unchanged N7_H projected artifact and
original snapshot for provenance hashing, VTK, Matplotlib, Poppler, sufficient
RAM, and write access to a new destination. It does not run implicitly in pytest.

P7 is a representation of the existing projected SEM field, not a new DNS
resolution or an accuracy proof. There is no numerical smoothing or filtering.
Only real frame 79 has currently been reconstructed by this workflow; there is
no high-order omega_x slice time series. The output is a dense 2D x-z slice,
not a full 3D high-order volume. C0 agreement at shared nodes does not imply
continuous derivatives; native field oscillations, SEM interfaces, overshoot,
and finite dense-grid rasterization may still affect appearance. Only this
validated, axis-aligned N7_H native P7 topology is supported by the real-data
runner; it is not a generic curvilinear or arbitrary-order VTR importer.
