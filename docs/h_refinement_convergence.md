# H-refinement convergence diagnostics

Run from the repository root after the completed Step 3 field comparison:

```bash
PYENV_VERSION=research312 python scripts/30_analyze_h_refinement_convergence.py
```

The command reads the stored `GC0.f00001` geometry through the project Nek reader
and existing GLL geometry validator. It reads Step 3 numerical fields and masks
from `/data/Nek5000_data/results/h_refinement/field_comparison/`, creates only
`/data/Nek5000_data/results/h_refinement/convergence_analysis/`, and compares SHA-256 hashes of
all Step 3 files before and after analysis. It refuses an existing output
directory. No raw data or Step 3 artifact is modified.

## Actual directional resolution

All elements are certified as complete axis-aligned affine tensor partitions
within the documented float32 coordinate storage tolerance. The physical to
reference-axis mapping is x→2, y→1, z→0 in all cases. Widths below are *physical
element widths*, not GLL node separations or the 500 × 200 post-processing grid
spacing. The mean weights each unique directional interval equally; the JSON
and CSV also include medians, length-weighted means, all interval endpoints,
and tolerance estimates.

| Case | Elements x × y × z | Δx min / max / mean | Δy min / max / mean | Δz min / max / mean / median |
| --- | --- | --- | --- | --- |
| N7_H | 272 × 12 × 8 | .125 / .125 / .125 | .125 / .125 / .125 | .086700 / .169400 / .125000 / .121950 |
| N7_VH | 380 × 17 × 10 | .08947277 / .08947372 / .08947368 | .08823526 / .08823538 / .08823529 | .055300 / .157900 / .100000 / .093400 |
| N7_VVH | 490 × 22 × 12 | .06938744 / .06938934 / .06938776 | .06818175 / .06818187 / .06818182 | .039200 / .145500 / .083333 / .076150 |

The x/y min–max spread at VH/VVH is dominated by float32 coordinate storage.
The z variation is physical mesh grading. At matched physical positions,
coarse/fine local element-width ratios are:

| Transition | x | y | z local min–max |
| --- | ---: | ---: | ---: |
| H → VH | 1.397059 | 1.416667 | 1.072831–1.567812 |
| VH → VVH | 1.289474 | 1.294118 | 1.084314–1.412574 |

All directions refine locally everywhere within storage tolerance; VVH is the
finest available mesh by actual physical element widths, not merely by element
count. Refinement is anisotropic and the z ratio changes with position. There
is no scientifically justified scalar h for this three-mesh family. The
vertical VVH grading is also mildly asymmetric beyond storage roundoff (for
example, the first two z widths are .0392, .0510 and the last two .0509,
.0393). The directional spacing distributions and local ratios are retained.

## Solution-difference criteria

For each field and target time, all three fields use exactly the Step 3 saved
physical y=.75 slice and identical 500 × 200 x-z coordinates. A single
field-specific mask intersects geometry-valid and finite values for all three
cases. The pairwise table reports absolute L2, RMS L2, mean absolute error,
and maximum absolute error for H−VH, VH−VVH, and H−VVH. Norms are unweighted
Cartesian grid-sample norms. The finest-reference errors are copied and
cross-checked against Step 3, with VVH identified as the *finest available
numerical reference*, never an exact solution.

Classification uses RMS norms D01=RMS(H−VH), D12=RMS(VH−VVH), and
D02=RMS(H−VVH). Let ρ=D12/D02, with a 5% diagnostic band:

- **Convergence-like:** ρ < .95 and nonnegative alignment of the two
  difference vectors, provided differences exceed an absolute 1e-14 threshold.
- **Non-monotonic:** ρ > 1.05, or the H−VH and VH−VVH difference vectors have
  negative dot product (oscillatory/cancelling changes).
- **Approximately unchanged / inconclusive:** the remaining cases, including
  ratios in [.95, 1.05] and negligible differences.

D12/D01 is recorded as a second diagnostic but cannot alone determine
monotonicity when successive mesh-size changes are unequal. The thresholds
are explicit diagnostic choices, not statistical confidence intervals.
Convergence-like means only that sampled solution differences meet these
rules; it does not establish an asymptotic error regime. Zero/tiny
finest-reference L2 denominator follows the established comparison-kernel
undefined-relative-error policy.

Observed order is computed only for manufactured synthetic inputs that have an
independently justified scalar h, sufficient difference separation and nearly
collinear same-signed difference vectors. Unequal spacing ratios solve the full
three-mesh power-law equation
`D01/D12 = (h0^p − h1^p)/(h1^p − h2^p)` numerically. The simple equal-ratio
log expression is not used for unequal ratios. All real fields have no observed
order because no scalar h is justified; moreover saved physical times differ
slightly across cases and there was no temporal interpolation. Three meshes
alone cannot prove an asymptotic regime.

## Phase and morphology diagnostics

The analysis also reports fixed-support integer x-shift searches on x<0 and
x>0 halves (up to .5 physical units, no wrapping/extrapolation), and RMS
distance between sorted field values on the identical mask. Shifts are never
applied to the reported pairwise or Step 3 errors. Sorted-value distance is
mathematically no greater than spatial RMS; a smaller value alone does not
prove phase divergence. Together with late-time changes, shift reductions may
make phase or morphology divergence plausible, but cannot establish its cause.
These are 2-D midspan diagnostics for evolving 3-D flows; no front/leading-edge
or Cantero analysis is performed. Time offsets and unverified physical/solver
conditions are additional limitations.

The conditional approach follows [NASA Glenn's grid-convergence tutorial](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html), which ties
observed order to an asymptotic leading-error model and distinguishes spatial
from temporal error. [NASA's discussion of chaotic eddy-resolving simulations](https://www.nas.nasa.gov/SC15/demos/demo7.html)
supports considering late-time phase sensitivity as a possibility; these
simulations have not been demonstrated to be chaotic by this analysis.

## Real-data results

The generated CSV and JSON files in `/data/Nek5000_data/results/h_refinement/convergence_analysis/`
contain the complete per-field/per-time pairwise values, classifications,
mesh intervals and local ratios. On the common finite mask, the RMS solution
differences are:

| Target time | Field | H−VH | VH−VVH | H−VVH | Classification |
| ---: | --- | ---: | ---: | ---: | --- |
| 5.0 | concentration | .01225079 | .01280244 | .01328351 | non-monotonic |
| 5.0 | velocity magnitude | .00376685 | .00436912 | .00474675 | non-monotonic |
| 5.0 | pressure fluctuation | .00191764 | .00253402 | .00271462 | non-monotonic |
| 10.0 | concentration | .04805650 | .05019614 | .05003026 | non-monotonic |
| 10.0 | velocity magnitude | .02543862 | .02600501 | .02653650 | non-monotonic |
| 10.0 | pressure fluctuation | .01192271 | .01406873 | .01415786 | non-monotonic |
| 15.0 | concentration | .07406427 | .07114996 | .07162137 | non-monotonic |
| 15.0 | velocity magnitude | .04763011 | .04706553 | .05129757 | non-monotonic |
| 15.0 | pressure fluctuation | .02171832 | .01895264 | .02286073 | non-monotonic |
| 19.5 | concentration | .11190326 | .08575900 | .11385752 | non-monotonic |
| 19.5 | velocity magnitude | .07724066 | .07267895 | .08742048 | non-monotonic |
| 19.5 | pressure fluctuation | .02930930 | .02789353 | .02983936 | non-monotonic |

Every row has 100,000/100,000 finite-valid points (fraction 1.0), with zero
NaNs and zero extrapolated points in the saved Step 3 diagnostics. The negative
successive-difference-vector cosine ranges from −.2808 to −.5141, triggering
the non-monotonic classification even when VH has a smaller finest-reference
RMS error than H. This describes oscillatory spatial changes between meshes;
it does not by itself establish divergence from the unknown exact solution.
All observed-order entries are blank. VVH has zero self-error by construction
and is not an independent convergence datapoint.

All three nearest-time selections happened to share file indices 21, 41, 61,
and 79 for targets 5.0, 10.0, 15.0, and 19.5, respectively. Selection used
stored physical times, not index equality. The largest absolute time error is
.00350711 (H at target 10.0), and the largest within-triplet time spread is
.00305767 (target 5.0). These time mismatches are an additional reason not to
fit a spatial order. The exact y=.75 plane coincides with stored GLL layers in
H and VVH; VH requires physical-to-reference spectral interpolation. All
three were evaluated at identical physical x-z target coordinates.

The late-time pairwise RMS differences generally increase. The bounded
translation diagnostic finds at most a 7.80% RMS reduction for one x-half at
target 15.0 and 2.99% at target 19.5, using one post-processing-grid interval
(.06814 physical x units); no shift-search boundary is hit. This makes modest
phase displacement a plausible contributor for some late-time slices, but it
cannot identify the dynamics or explain all differences. The sorted-value
diagnostic is deliberately not treated as independent proof of phase change.
No shifted field is used in any reported error.

The analysis records SHA-256 digests for all 45 saved Step 3 files; hashes
were unchanged across the analysis run. Machine-readable details are in
`analysis.json`, `pairwise_differences.csv`,
`convergence_classification.csv`, `directional_mesh_resolution.csv`,
`local_refinement_ratios.csv`, `finest_reference_errors.csv`,
`morphology_diagnostics.csv`, and `translation_diagnostics.csv`. The two
figures are `pairwise_differences.png` and `directional_mesh_spacing.png`.
