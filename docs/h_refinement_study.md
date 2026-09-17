# H-refinement study: final scientific and software audit

This page records the completed N=7 study on `feature/h-refinement` using the
saved Step 1–6B artifacts. It separates instantaneous midspan fields, the
span-averaged Cantero mean front, and local leading-edge morphology. All
comparisons to `N7_VVH` use it as the **finest available numerical reference**,
never as an exact solution. The p-refinement configuration and result tree are
separate. The detailed methods and full tables remain in the [inventory and
field study](h_refinement.md), [directional analysis](h_refinement_convergence.md),
[Cantero study](h_refinement_cantero.md), and [leading-edge study](h_refinement_leading_edge.md).

## Data, mesh, and configuration

The cases are configured in `config/h_refinement.yaml`, independently of the
N5/N7/N9/N11 polynomial-order study in `config/cases.yaml`. The raw directories
come from `config/paths.yaml`. Fresh inspection of all 243 Nek headers finds
81 `GC0.f00001`–`GC0.f00081` files per case, strictly increasing stored times
from 0 to 20, 8 × 8 × 8 GLL nodes per element (N=7), and identical `XUPT`
variable availability: coordinates, three velocities, pressure, and one `T`
storage slot used as concentration by the project accessor. Header codes alone
do not establish the physical meaning of that slot. The representative meshes
have the same certified box topology and physical bounds:
`x=[-17,17]`, `y=[0,1.5]`, `z=[0,1]`.

| Case | Raw directory | Elements x × y × z | Total elements | Mean element Δx / Δy / Δz | Physical Δz min–max |
| --- | --- | ---: | ---: | --- | --- |
| N7_H | `/data/Nek5000_data/case_N7_H` | 272 × 12 × 8 | 26,112 | .125000 / .125000 / .125000 | .0867–.1694 |
| N7_VH | `/data/Nek5000_data/case_N7_VH` | 380 × 17 × 10 | 64,600 | .08947368 / .08823529 / .100000 | .0553–.1579 |
| N7_VVH | `/data/Nek5000_data/case_N7_VVH` | 490 × 22 × 12 | 129,360 | .06938776 / .06818182 / .08333333 | .0392–.1455 |

These are **physical element widths**, not GLL-node or post-processing-grid
spacings. Matched-position H→VH local ratios are approximately 1.3971 in x,
1.4167 in y, and 1.0728–1.5678 in z; VH→VVH ratios are approximately
1.2895, 1.2941, and 1.0843–1.4126. The z meshes are graded. Refinement is
anisotropic, so one scalar characteristic h and a global observed order are
not justified. VVH is finest by verified local directional widths, not merely
total element count.

## Comparison coordinates and physical time

| Study | Common physical samples | Time policy | Finite coverage |
| --- | --- | --- | --- |
| Steps 2–4, midspan fields | Exact `y=0.75`; `x_j=-17+34j/499` for j=0…499; `z_k=k/199` for k=0…199; 500 × 200 Cartesian targets | Independently select nearest *stored physical time* to 5, 10, 15, 19.5; no temporal interpolation | 100,000/100,000 per field and target |
| Step 5, Cantero mean front | Native GLL quadrature in physical z and y, then a one-dimensional threshold front | Each case's 81 actual stored times for velocity/reconstruction; reference and paper comparisons interpolate only inside shared ranges | 81/81 successful frames per case |
| Steps 6A–6B, leading edge | `z=0.04`; `x_j=-17+34j/999` for j=0…999; periodic `y_k=1.5k/308` for k=0…307; upper y endpoint excluded | Preserve original curves; align H/VH linearly to VVH's 81 stored times in the common interval [0,20] | 308/308 y crossings per case and frame |

The 500 × 200 and 1000 × 308 grids are fixed **post-processing sampling**
grids; neither is a simulation mesh or a grid-convergence study. Spectral
element inversion evaluates the requested physical planes directly. Targets
without an element owner remain invalid; no spatial extrapolation or
nearest-plane replacement is used. H and VVH contain a stored GLL `y=0.75`
layer, while VH requires physical-to-reference interpolation. All four Step 3
snapshots have identical saved x-z coordinate arrays, zero NaNs, zero inverse-map
failures and zero extrapolated targets. The largest Step 3 target-time offset is
.00350711 and the largest three-case time spread is .00305767; the field
differences therefore include small temporal offsets. Step 6B's largest
same-index time spread is .00488286, which is why its curves are aligned by
physical time. Its 308,000 plane targets map in every case, and its aligned
all-case finite mask covers all 308 y positions at every time. No temporal
extrapolation or failed-crossing fill is allowed.

## Instantaneous fields and directional differences

Concentration is the established concentration accessor, velocity magnitude
is formed *after* interpolating its three components, and pressure fluctuation
is each case's sampled pressure minus its arithmetic mean on the **all-case
finite-pressure mask**. Step 3 reports relative L2, mean absolute, and maximum
absolute errors on one field-specific all-case geometry-and-finite mask. VVH's
zero self-error is a control, not an independent convergence datapoint. The
following relative L2 values summarize H/VH differences from VVH; the full
MAE, maximum, time, and mask columns are in
`/data/Nek5000_data/results/h_refinement/field_comparison/field_errors.csv`.

| Target time | Field | H relative L2 | VH relative L2 |
| ---: | --- | ---: | ---: |
| 5 | Concentration | .01960 | .01889 |
| 5 | Velocity magnitude | .02068 | .01903 |
| 5 | Pressure fluctuation | .01291 | .01206 |
| 10 | Concentration | .07513 | .07537 |
| 10 | Velocity magnitude | .08568 | .08397 |
| 10 | Pressure fluctuation | .06959 | .06915 |
| 15 | Concentration | .11038 | .10965 |
| 15 | Velocity magnitude | .14066 | .12906 |
| 15 | Pressure fluctuation | .12214 | .10126 |
| 19.5 | Concentration | .17937 | .13510 |
| 19.5 | Velocity magnitude | .23385 | .19442 |
| 19.5 | Pressure fluctuation | .17293 | .16165 |

Step 4 also records H−VH, VH−VVH, and H−VVH norms on those same saved
coordinates and masks. All 12 field/time triplets meet its explicit
**non-monotonic** diagnostic criterion because successive difference vectors
have negative alignment. For example, concentration RMS pairwise differences
at target 10 are .04806, .05020 and .05003, respectively. Classification is
about these sampled numerical solutions, not an exact-solution error trend.
No real-data observed order is reported. Synthetic equal- and unequal-ratio
order checks validate the conditional solver only where its assumptions hold.

## Cantero mean front and leading-edge morphology

Step 5 reuses the established Re=3450 numerical path: physical GLL integration
in z, spanwise averaging in y, first positive-x downward mean-height crossing
at `delta=0.01` from `reference_x=0`, `numpy.gradient` at the case's actual
times, 11-point moving-average velocity smoothing, and trapezoidal trajectory
reconstruction. There are no failed frames. Comparisons to digitized Cantero
Figure 5a use 34 paper times in the overlap `0.56721–17.318`; the literature
curve is not an exact numerical solution.

| Case | Final raw mean front | Reconstructed displacement | RMS vs Cantero paper | RMS raw front vs VVH | RMS reconstructed front vs VVH |
| --- | ---: | ---: | ---: | ---: | ---: |
| N7_H | 16.10495 | 8.08986 | .06095 | .02114 | .02017 |
| N7_VH | 16.17271 | 8.14028 | .05615 | .03835 | .02672 |
| N7_VVH | 16.14787 | 8.12999 | .05523 | 0 by definition | 0 by definition |

These bulk trajectories are relatively close but do not decrease monotonically
in difference from VVH as element count increases. Their integral threshold
definition is distinct from the instantaneous midspan fields and from the
leading edge at `z=0.04`.

Step 6B uses concentration threshold 0.1 and strictly positive x. Both the
primary rightmost-crossing method and independent Moore-boundary method have
24,948/24,948 successful y crossings per case across 81 frames. Their front
positions and success masks are exactly identical in all 243 case-frames:
per-frame method RMS and maximum differences are zero, with no differing y
locations. Candidate crossing counts can differ because the methods count
candidates differently; this is not a front-position difference. No smoothing
is applied to leading-edge curves.

At aligned time 20, the H/VH total spanwise RMS differences from VVH are
.07532/.08920, signed mean-front differences are −.02971/+.04526, and
mean-removed shape RMS differences are .06921/.07687. VH is closer than H to
VVH at only 6/81 times in total RMS and 8/81 in shape RMS. At time 20, the
spanwise standard deviations are .03984/.04045/.06587 for H/VH/VVH, and
peak-to-peak amplitudes are .18381/.16644/.25997. Thus VVH shows greater
late-time sampled spanwise variation; that does not establish that its
lobe-cleft amplitude is more physically correct.

## Conclusions and limits

The data support: common N=7 and box-domain geometry; strictly finer measured
directional element widths from H to VH to VVH; fully covered comparisons on
the stated physical grids; non-monotonic midspan solution differences; close
but non-monotonic bulk Cantero propagation; resolution-sensitive local
spanwise morphology; and zero observed front-position sensitivity to the two
extraction methods on these 243 case-frames.

The data do **not** establish formal asymptotic h-convergence, a global observed
order, an exact VVH solution, chaotic dynamics, or greater physical accuracy
of VVH's late-time lobe-cleft amplitude. The initial/boundary conditions,
physical parameters and solver settings have not been independently certified
equal. Field snapshots differ slightly in time; Step 6B's linear temporal
interpolation adds an approximation. Temporal discretization error and
post-processing-grid sensitivity were not quantified. Late-time translation
and morphology diagnostics make phase/structure divergence plausible, not a
demonstrated cause of any difference.

## Reproduction and artifacts

The saved outputs are under `/data/Nek5000_data/results/h_refinement/`: `field_comparison/`,
`convergence_analysis/`, `cantero_mean_front/`, `cantero_re3450/`, and
`leading_edge/step6a_smoke/` and `leading_edge/step6b/`. Raw Step 6B histories
remain separate from aligned derived tables. The canonical root is configured
as `h_refinement_results_root` in `config/paths.yaml` and is outside Git;
the obsolete repository-local `results/h_refinement/` tree has been removed.
The path-only migration preserved all 364 generated files (74,836,984 bytes)
byte for byte, verified by per-file SHA256. Two historical Cantero CSV summaries
retain their original `front_source` strings, and two Step 6B logs retain their
original output strings; these are provenance from the original run, not current
reader defaults. They were left unchanged to preserve the validated artifacts.
These producer commands are for a **fresh** subdirectory under the canonical
root because the scripts refuse existing outputs; the current
validated artifacts do not need to be recomputed for review. The full Step 6B
extraction reads all 243 large Nek snapshots and is correspondingly expensive.

```bash
export MPLCONFIGDIR=/tmp/matplotlib-nek-post
audit_root=/data/Nek5000_data/results/h_refinement/reproduction_NEW_ID  # choose a fresh name

PYENV_VERSION=research312 python scripts/27_inventory_h_refinement.py
PYENV_VERSION=research312 python scripts/28_extract_h_refinement_slice.py \
  --snapshot N7_H=81,N7_VH=81,N7_VVH=81 --nx 101 --nz 41 \
  --output-dir "$audit_root/physical_slice_smoke"
PYENV_VERSION=research312 python scripts/29_compare_h_refinement_fields.py \
  --nx 500 --nz 200 --max-time-error 0.01 --max-time-spread 0.01 \
  --output-dir "$audit_root/field_comparison"
PYENV_VERSION=research312 python scripts/30_analyze_h_refinement_convergence.py \
  --input-dir "$audit_root/field_comparison" \
  --output-dir "$audit_root/convergence_analysis"
for case in N7_H N7_VH N7_VVH; do
  PYENV_VERSION=research312 python scripts/24_compute_cantero_mean_front.py \
    --case "$case" --output-dir "$audit_root/cantero_mean_front" \
    --stationary-geometry-fast-path
done
PYENV_VERSION=research312 python scripts/26_overlay_cantero_re3450_multicase.py \
  --cases N7_H N7_VH N7_VVH \
  --front-root "$audit_root/cantero_mean_front" \
  --output-dir "$audit_root/cantero_re3450" \
  --reference-case N7_VVH --include-input-summary
PYENV_VERSION=research312 python scripts/31_compare_h_refinement_leading_edges.py \
  --output-dir "$audit_root/leading_edge/step6b"
```

For code-only regression without producing new scientific artifacts:

```bash
PYENV_VERSION=research312 python -m pytest -q tests/test_h_refinement*.py \
  tests/test_leading_edge_comparison*.py
PYENV_VERSION=research312 python -m pytest -q
git diff --check
git ls-files results/h_refinement
git check-ignore -v results/h_refinement/field_comparison/run.json
```
