# H-refinement mesh/data inventory

The h-study is configured independently in `config/h_refinement.yaml`. Its case
sequence is a candidate coarse-to-fine ordering, its expected polynomial order
is an assertion to test, and `N7_VVH` is a provisional reference. Directories are
resolved through `ProjectPaths` and `config/paths.yaml`. The p-study's
`config/cases.yaml`, `orders`, N11 reference, comparison sets, and output roots
are unchanged.

Run from the repository root:

```bash
PYENV_VERSION=research312 python scripts/27_inventory_h_refinement.py
# Optional report capture outside the existing p-study output tree:
PYENV_VERSION=research312 python scripts/27_inventory_h_refinement.py > /tmp/nek_h_inventory.json
```

The command prints JSON and creates no files itself. `--paths-config` and
`--study-config` select alternative configuration files. Scientific inventory
and validation live in `nek_post.h_refinement`; the script handles arguments
and serialization only. A false validation flag is a successful inventory
result, not a CLI error; missing/unreadable or unsupported data exits with an
error.

The inventory reuses exact `PREFIX.fNNNNN` discovery and the project's Nek5000
reader. A header-only adapter in `io_nek` delegates to pymech's header reader.
Every field header is scanned for time, GLL dimensions, element count, and
variable availability. One coordinate-bearing field is read per case, skipping
solution-variable I/O, and all its elements contribute to the domain bounds.
Pymech still allocates payload arrays; large cases can require several GB of RAM.
Cases are processed sequentially.

The JSON includes every discovered filename and physical time, first/last
indices, count, time range, output-spacing min/median/max and uniformity,
representative mesh source, scalar and coordinate array shapes, per-axis GLL
node counts, total element-local GLL nodes, polynomial orders, element count,
x/y/z extents, variable codes/counts, and warnings. Variable-count vectors are
ordered as coordinates, velocity, pressure, temperature, passive scalars.
`temperature` names the Nek storage slot; the header alone does not establish
whether it represents physical temperature or concentration.

Polynomial degree is nodes-per-active-axis minus one. Anisotropic degrees are
reported per axis with no single scalar degree; 2D singleton z is excluded from
the degree calculation. Split-file dumps, nonfinite times/coordinates,
missing coordinates, and changing mesh order/count across headers are rejected.
Index gaps and duplicate/reversed physical times are reported; non-increasing
times suppress the representative spacing. Output spacing is not the solver's
internal timestep. Coordinates are assumed stationary over the series.

## Real-data inspection, 2026-09-16

All 243 headers were scanned. Each case has 81 files, `GC0.f00001` through
`GC0.f00081`, covering physical time 0 through 20 without index gaps. Each uses
8 x 8 x 8 GLL nodes (512 element-local nodes), polynomial degree N=7, scalar
array shape `(8, 8, 8)` in `(lz, ly, lx)` order, and coordinate shape
`(3, 8, 8, 8)`. Element counts and orders remain constant across each series.

| Case | Data directory | Elements | Median output spacing | Min / max spacing |
| --- | --- | ---: | ---: | --- |
| N7_H | `/data/Nek5000_data/case_N7_H` | 26,112 | 0.250333060 | 0.245941252 / 0.255182945 |
| N7_VH | `/data/Nek5000_data/case_N7_VH` | 64,600 | 0.249824948 | 0.246823747 / 0.255800202 |
| N7_VVH | `/data/Nek5000_data/case_N7_VVH` | 129,360 | 0.250426757 | 0.247256798 / 0.253355443 |

All representative meshes (`GC0.f00001`) have identical bounds:
`x = [-17, 17]`, `y = [0, 1.5]`, `z = [0, 1]`. The configurable absolute
comparison tolerance is `1e-6`, with relative tolerance zero. All headers carry
`XUPT`: coordinates, three velocity components, pressure, and one temperature
slot; no passive scalar slots. Output times are strictly increasing but
nonuniform and differ between cases; equal file indices must not be treated as
exactly aligned times in future work.

The metadata confirms the same degree and bounding domain, with strictly
increasing element counts H -> VH -> VVH. This is consistent with h-refinement,
and VVH is the unique largest mesh by element count. A fully controlled pure
h-refinement study is not established by this inventory alone: identical
boundary/initial conditions, physical parameters, solver settings, and domain
topology have not been verified. Equal bounds do not prove equal interior
domains, and largest element count does not prove finest spacing everywhere or
convergence. Mesh coordinate invariance over time is assumed. No field,
front, or leading-edge comparisons are implemented.

## Exact physical midspan workflow

`scripts/28_extract_h_refinement_slice.py` evaluates a common **physical**
`y = 0.75` plane. It never selects a nearest stored plane. The library entry
point is `nek_post.h_refinement_slice.sample_h_snapshot`; call it for each
selected group of case indices. The CLI supports repeated `--snapshot` groups.

```bash
PYENV_VERSION=research312 python scripts/28_extract_h_refinement_slice.py \
  --snapshot N7_H=81,N7_VH=81,N7_VVH=81 \
  --nx 101 --nz 41 --output-dir /tmp/nek_h_slice_t20
```

All configured study cases must be present in every group. Headers must contain
coordinates and all required fields, and match the study's expected 3D order.
Split-file dumps are rejected. Physical times must agree within `--time-atol`
(default `1e-8`); identical file indices alone do not establish time alignment.
There is no time interpolation. A deliberately relaxed tolerance is recorded
along with every actual time and the resulting time spread.

### Numerical method and field definitions

The existing `build_spectral_slice_interpolation_plan` already supports an exact
physical y-plane via damped Newton inversion of each candidate element's
isoparametric map. Its additive `target_grid` argument now accepts the same
explicit x-z coordinates for every case. The tensor-product GLL barycentric
basis evaluates solution values at the recovered reference coordinates.
Reference-axis orientation is inferred through the physical map, rather than
assuming a particular array axis represents physical y.

Plane restriction and final Cartesian-grid sampling are conceptually distinct:

1. Restrict the continuous element interpolant to physical `y=0.75` (within the
   inverse-map numerical tolerance).
2. Sample that restriction at the shared Cartesian `(Xi, Zi)` targets.

These operations are evaluated together directly from the 3D interpolant.
There is **no second scattered/linear interpolation**, no projection of nearby
y-planes, and no nearest-neighbor fill. The existing p-refinement nearest-plane
and scattered-grid workflow retains its defaults.

The real-data validation grid used 101 x coordinates and 41 z coordinates,
including endpoints, with output shape `(nz, nx)`. For these datasets it spans
`x=[-17,17]`, `z=[0,1]`, so `dx=0.34`, `dz=0.025`. `--nx` and `--nz` are required explicitly (also required by the library API);
there is no default production grid. The validation resolution is not a
convergence-qualified grid.
Full x/y/z bounding domains must match within the configured absolute tolerance
(`1e-6`, relative tolerance zero). The grid uses the intersection of the accepted
bounds, preventing small tolerated boundary differences from extending targets
outside the shared box. Equal boxes alone do not establish equal topology;
unmapped targets, including holes, remain NaN with false geometry masks.

- `C`: the existing `temp[0]`, falling back to `scal[0]`, accessor.
- `u`, `v`, `w`: interpolated separately; `speed = sqrt(u²+v²+w²)` afterwards.
- `p_prime`: interpolated pressure minus each case's arithmetic mean on the
  all-case finite-pressure mask, as in existing pressure comparisons. This is
  a Cartesian sample mean, not a volume-weighted mean. The established project
  helper is reused, including conversion of nonfinite raw pressure to NaN in p′.

At duplicated element interfaces, the existing plan assigns one owner by
minimum inverse-map residual, then element index. Nodes are not pooled across
elements, so duplication does not weight means or add target samples. This
policy chooses one trace if fields are discontinuous; it does not average
traces. Interface ambiguity counts are reported. Inverse mapping uses physical
tolerance `1e-11` times element scale and reference tolerance `1e-8`; only
accepted reference points in the element (with tolerance-level boundary
clamping) are evaluated. Unsupported/unmapped targets are never extrapolated.

### Artifacts and real-data validation

The CLI requires an explicit output directory outside raw case directories and
refuses existing artifact names. It writes `snapshot_001.npz` and
`snapshot_001.json` (incrementing for additional groups). The NPZ contains `Xi`,
`Zi`, `y_target`, `<case>_C/u/v/w/speed/p/p_prime`, per-case geometry masks, and
common masks for C, speed, p, geometry and all fields. Arrays load with
`allow_pickle=False`. The JSON records file provenance, actual times, grid,
stored-plane diagnostics, inverse-map diagnostics, per-field NaN/finite
fractions, common valid fractions, and pressure means. Apply the appropriate
common mask before any future comparison. No convergence metrics are computed.

A real-data smoke run at `f00081`, physical `t=20` in all three cases, used the
101 x 41 grid (4,141 points):

| Case | Exact stored y=0.75 layer | Nearest stored y distance | Intersecting elements | Maximum inverse residual |
| --- | --- | ---: | ---: | ---: |
| N7_H | Yes, in all intersecting elements | 0 | 4,352 | 5.69e-14 |
| N7_VH | No; interpolation required | 0.00923377275466919 | 3,800 | 1.04e-13 |
| N7_VVH | Yes, in all intersecting elements | 0 | 11,760 | 9.98e-13 |

Exact stored-layer detection checks every coordinate on an entire reference
layer for exact equality, without rounding. Even when such a layer exists,
x/z targets generally still require interpolation. This diagnosis is performed
for each selected snapshot, not assumed from the case label or an earlier mesh.

All three cases had 100% geometry and finite-field coverage, 0% NaNs in every
output field, zero inverse failures and zero extrapolated points. Shared masks
also covered 100%. Ambiguous interface targets numbered 4,141, 221 and 4,141,
respectively; these were resolved by the documented ownership rule. This
validates the sampling workflow, not h-convergence or grid-resolution adequacy.
