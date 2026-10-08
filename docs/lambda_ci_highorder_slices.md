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
