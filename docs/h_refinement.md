# H-refinement mesh/data inventory

The h-study is configured independently in `config/h_refinement.yaml`. Its case
sequence was initially a candidate coarse-to-fine ordering, its expected
polynomial order is an assertion to test, and `N7_VVH` was initially a
provisional reference. Directories are
resolved through `ProjectPaths` and `config/paths.yaml`. The p-study's
`config/cases.yaml`, `orders`, N11 reference, comparison sets, and output roots
are unchanged.

Run from the repository root:

```bash
PYENV_VERSION=research312 python scripts/27_inventory_h_refinement.py
# Optional report capture in a fresh h-study results subdirectory:
mkdir -p /data/Nek5000_data/results/h_refinement/reproduction_NEW_ID
PYENV_VERSION=research312 python scripts/27_inventory_h_refinement.py > /data/Nek5000_data/results/h_refinement/reproduction_NEW_ID/inventory.json
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
exactly aligned times in downstream comparisons.

The metadata confirms the same degree and bounding domain, with strictly
increasing element counts H -> VH -> VVH. This is consistent with h-refinement,
and VVH is the unique largest mesh by element count. A fully controlled pure
h-refinement study is not established by this inventory alone: identical
boundary/initial conditions, physical parameters, solver settings, and domain
topology have not been verified. Equal bounds do not prove equal interior
domains, and largest element count does not prove finest spacing everywhere or
convergence. The inventory itself assumes mesh coordinate invariance over time
and performs no field, front, or leading-edge comparisons.

## Exact physical midspan workflow

`scripts/28_extract_h_refinement_slice.py` evaluates a common **physical**
`y = 0.75` plane. It never selects a nearest stored plane. The library entry
point is `nek_post.h_refinement_slice.sample_h_snapshot`; call it for each
selected group of case indices. The CLI supports repeated `--snapshot` groups.

```bash
PYENV_VERSION=research312 python scripts/28_extract_h_refinement_slice.py \
  --snapshot N7_H=81,N7_VH=81,N7_VVH=81 \
  --nx 101 --nz 41 --output-dir /data/Nek5000_data/results/h_refinement/reproduction_NEW_ID/physical_slice_smoke
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

## Field comparison by physical snapshot time

`scripts/29_compare_h_refinement_fields.py` compares C, velocity magnitude, and
pressure fluctuation using the exact-plane workflow above. It first runs the
mesh inventory and requires common polynomial order/domain bounds and a unique
largest-element-count reference. Element counts alone did not establish local
spacing; [Step 4](h_refinement_convergence.md) subsequently verified VVH as
finest by directional element widths. Identical simulation physics remains
unverified.

```bash
PYENV_VERSION=research312 python scripts/29_compare_h_refinement_fields.py \
  --nx 500 --nz 200 \
  --max-time-error 0.01 --max-time-spread 0.01 \
  --output-dir /data/Nek5000_data/results/h_refinement/field_comparison
```

The grid dimensions remain mandatory. The executed comparison uses the
user-selected 500 x 200 grid, matching the p-workflow's configured dimensions,
with x in [-17,17], z in [0,1], and physical y=0.75. The same coordinates are
required across all three cases and all selected target times. This resolution
has not itself been shown to give grid-independent comparison errors. It is a
fixed post-processing comparison grid, not the simulation mesh resolution and
not evidence of grid convergence.

By default, target times come from `multitime_comparison_sets` and their
`target_time` entries in `config/cases.yaml`: 5, 10, 15, 19.5. Only those time
conventions are borrowed; p-case membership, polynomial orders, reference, and
artifacts are not used. `--times` can select another explicit target list.
For every case, discovery and the existing Nek header reader supply all actual
physical times. The nearest timestamp is selected independently; ties choose
the earlier timestamp, then the lower index. Out-of-range targets, nonfinite
times, and selections exceeding the target-error or group-spread limits are
rejected. Both limits default to 0.01 and are recorded in `run.json`. There is
no temporal interpolation: errors therefore include any effect of the recorded
small snapshot timing offsets and must not be interpreted as pure spatial
errors at exactly identical times.

The additive `compare_sampled_grids` kernel in `nek_post.comparison` compares
already co-located scalar grids. It reuses the established metric functions and
does not alter the existing p-refinement kernels. For each field, one mask is
the intersection of geometry-valid targets across **all three cases** and
finite values of that field across **all three cases**. Every case, including
the reference control, uses that same mask. These are per-field masks, not
pairwise masks and not a forced intersection across unrelated fields.
Pressure fluctuation retains the established per-case arithmetic pressure mean
over the all-case finite-pressure mask. The normal sampling workflow ensures
unmapped pressure targets are NaN.

Reported metrics use unweighted Cartesian samples, as in the p-workflow:

- relative L2 = sqrt(sum((value-reference)^2) / sum(reference^2));
- mean absolute error = mean(abs(value-reference));
- maximum absolute error = max(abs(value-reference)).

As in the established safe relative metric helper, reference squared norm
<= 1e-14 produces an undefined (`nan`) non-reference relative L2, flagged by
`relative_l2_defined=False`. Absolute errors remain defined. Reference rows
are explicit controls with zero self-error and `independent_datapoint=False`;
they are excluded from error-history plots. No observed convergence order or
mesh-size refinement ratio is inferred.

The output directory must be new, contain `h_refinement` as a path component,
and be outside raw case directories and `poly_order_compare`:

- `inventory.json`: measured mesh inventory and reference validation evidence.
- `run.json`: grid, targets, time tolerances, reference basis, and completion status.
- `selected_times.csv`: target/actual time, index, signed/absolute time error,
  and source filename for every case and target.
- `field_errors.csv`: all three metrics, reference/control labels, mesh element
  counts, target/actual/reference times, grid bounds/resolution, mask counts and
  valid fractions (36 rows for four targets, three fields, three cases).
- `error_history.png`: a field-by-metric panel of non-reference errors against
  target physical time; no polynomial-order axis.
- `snapshot_001/` through `snapshot_004/`: per-target CSV tables, JSON diagnostics,
  numerical grids/components/fields and masks in `comparison_grids.npz`, and
  six absolute-difference contour plots (two non-reference cases x three fields).

Archives load with `allow_pickle=False`. JSON metadata includes exact-plane
coverage and per-field NaN fractions. Existing artifacts are never overwritten.
An interrupted run retains `status=running`, distinguishing it from a complete
result set; choose a new directory for a rerun.

For the 500 x 200 run, the header-based nearest selections are:

| Target | Index (all cases, independently selected) | N7_H time | N7_VH time | N7_VVH time |
| --- | ---: | ---: | ---: | ---: |
| 5.0 | 21 | 5.000312341405 | 5.003370008996 | 5.002819980228 |
| 10.0 | 41 | 10.003507107900 | 10.001046787750 | 10.001532502540 |
| 15.0 | 61 | 15.001384683450 | 15.000334524640 | 15.001537138340 |
| 19.5 | 79 | 19.502404088070 | 19.501780007520 | 19.500517542740 |

The equal selected indices are a result of the header-time search, not an
assumption. Maximum absolute target offset is 0.003507107900; maximum group
spread is 0.003057667591. Machine-readable selections retain full precision.

### Completed Step 3 real-data validation

All four target groups completed on the identical 500 x 200 physical x-z grid.
Every field at every target has 100,000/100,000 common valid points (fraction
1.0). Every case has 0% field NaNs, zero inverse-map failures, and zero
extrapolated targets. All 108 metric values were independently recomputed from
saved arrays/masks and matched the CSV values exactly. Aggregate and per-target
CSV rows agree, and selected times were checked against the raw headers.

The table below summarizes all non-reference metrics; CSVs retain full precision.
Every N7_VVH control row has zero self-error and is excluded from the plotted
independent case series.

| Target | Case | Field | Relative L2 | Mean absolute error | Maximum absolute error |
| --- | --- | --- | ---: | ---: | ---: |
| 5.0 | N7_H | concentration | 0.01959889 | 0.002320535 | 0.4538997 |
| 5.0 | N7_VH | concentration | 0.01888911 | 0.002257875 | 0.2656694 |
| 5.0 | N7_H | velocity magnitude | 0.02067503 | 0.001403914 | 0.08798031 |
| 5.0 | N7_VH | velocity magnitude | 0.0190302 | 0.001604476 | 0.06636532 |
| 5.0 | N7_H | pressure fluctuation | 0.01291454 | 0.0008905187 | 0.04932657 |
| 5.0 | N7_VH | pressure fluctuation | 0.01205539 | 0.0008806743 | 0.04061313 |
| 10.0 | N7_H | concentration | 0.07512522 | 0.01311681 | 0.7253162 |
| 10.0 | N7_VH | concentration | 0.07537431 | 0.01092378 | 0.8464996 |
| 10.0 | N7_H | velocity magnitude | 0.08568353 | 0.01088282 | 0.2921194 |
| 10.0 | N7_VH | velocity magnitude | 0.08396739 | 0.009642497 | 0.3646947 |
| 10.0 | N7_H | pressure fluctuation | 0.06958632 | 0.005502056 | 0.1821931 |
| 10.0 | N7_VH | pressure fluctuation | 0.06914823 | 0.005061054 | 0.1552397 |
| 15.0 | N7_H | concentration | 0.1103799 | 0.02424544 | 0.8802945 |
| 15.0 | N7_VH | concentration | 0.1096534 | 0.02229699 | 0.9553662 |
| 15.0 | N7_H | velocity magnitude | 0.1406609 | 0.02513713 | 0.4179674 |
| 15.0 | N7_VH | velocity magnitude | 0.1290564 | 0.02231781 | 0.4076394 |
| 15.0 | N7_H | pressure fluctuation | 0.1221434 | 0.01082584 | 0.2241079 |
| 15.0 | N7_VH | pressure fluctuation | 0.1012628 | 0.009334111 | 0.2022303 |
| 19.5 | N7_H | concentration | 0.179369 | 0.04565821 | 0.9593658 |
| 19.5 | N7_VH | concentration | 0.1351031 | 0.03404606 | 0.85719 |
| 19.5 | N7_H | velocity magnitude | 0.233851 | 0.04784867 | 0.6048848 |
| 19.5 | N7_VH | velocity magnitude | 0.1944172 | 0.03893827 | 0.6872938 |
| 19.5 | N7_H | pressure fluctuation | 0.1729274 | 0.01611294 | 0.3030742 |
| 19.5 | N7_VH | pressure fluctuation | 0.1616507 | 0.01400353 | 0.3206982 |

Errors do not decrease monotonically from H to VH for every metric and target
(for example, concentration relative L2 near t=10 and several maximum errors).
These outputs establish field discrepancies relative to the finest available
numerical mesh, not an observed h-convergence order. Snapshot timing offsets,
unverified physics equivalence, and the untested sensitivity to post-processing
grid resolution remain limitations. Directional mesh-size ratios and their
interpretation are reported in [Step 4](h_refinement_convergence.md); the
completed study is summarized in [the final audit](h_refinement_study.md).
