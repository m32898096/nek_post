# H-refinement leading-edge common grid (Step 6A)

The H, VH and VVH leading edges use the existing uniform-spectral interpolation
and extraction kernels on the physical plane `z_target=0.04`. Their common
post-processing grid has `nx=1000` x points and `ny=308` endpoint-excluded
periodic y points. The x coordinates are
`linspace(x_min, x_max, 1000)` including both physical domain bounds; the y
coordinates are `linspace(y_min, y_max, 308, endpoint=False)`. Matching physical
bounds must be verified before comparing cases. Each point is inverse-mapped
into a source spectral element and the original N=7 polynomial is evaluated
there. Low-level interpolation leaves unmapped points NaN; no extrapolation is
performed. The explicit-`ny` leading-edge workflow rejects any such incomplete
grid before extracting curves.

`ny=308` is a **sampling count**, chosen as twice the finest case's native
periodic-y unique count. It is not a simulation resolution or evidence of new
physical modes. The native count is measured from the deduplicated element GLL
coordinates, with the periodic upper seam excluded. The grid must be checked
for identical x and y coordinate arrays across the cases and full inverse-map
coverage before production comparison.

`scripts/19_compute_leading_edge_evolution.py --ny 308 --nx 1000` selects this
explicit-y route. `--ny` overrides the configured `y_upsample_factor`; when
`--ny` is omitted, the legacy N7 factor and CSV artifact behavior are
unchanged. The explicit route writes the existing timeseries CSV columns plus
tagged `*_leading_edge_sampling_metadata.json`, since a single integer
upsampling factor cannot describe all three source meshes on a shared 308-point
grid. Its default output tree is `/data/Nek5000_data/results/h_refinement/leading_edge/METHOD/CASE`.
Script 20 reads only the legacy factor-based metadata CSV; it is not a reader
for explicit-y artifacts.

The leading-edge definition remains `C=0.1`, strictly `x>0`, periodic y,
rightmost crossing. The independent Moore boundary extractor uses the same
sampled concentration plane for validation. Step 6A validates only a first
and a late frame in each case. It does not produce full evolution statistics
or h-convergence metrics.

## Real-data smoke validation

The first (`GC0.f00001`) and one late (`GC0.f00079`) field file were sampled
for each case. The measured physical bounds were exactly `x=[-17,17]`,
`y=[0,1.5]`; `z_target=0.04` was used throughout. The target x and y arrays
were compared with exact array equality across H, VH and VVH. Thus
`Δx=34/999` and `Δy=1.5/308`, with the last sampled y equal to
`1.49512987012987` and the periodic upper endpoint omitted.

| Case | Native periodic-y count | Elements | Late time at index 79 | Mapped targets | Inverse-map failures | Shared-interface ambiguous targets | Maximum successful residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N7_H | 84 | 26,112 | 19.50240408807 | 308,000 / 308,000 | 0 | 3,000 | 8.53e-14 |
| N7_VH | 119 | 64,600 | 19.50178000752 | 308,000 / 308,000 | 0 | 0 | 7.11e-14 |
| N7_VVH | 154 | 129,360 | 19.50051754274 | 308,000 / 308,000 | 0 | 1,000 | 1.17e-13 |

Index 1 has stored time 0.0 in all three cases. Both tested frames have zero
plane NaNs and 308 / 308 successful spanwise front positions per method in
every case. Rightmost crossing and Moore agree exactly on all six curves,
including success masks. Multiple inverse-map owners at shared element
interfaces account for the ambiguous-target counts; the interpolator selected
one owner deterministically. No target required extrapolation.

The smoke artifacts are in
`/data/Nek5000_data/results/h_refinement/leading_edge/step6a_smoke/`: `report.json` and twelve
small NPZ curve files, one per case, frame and method. They contain only the
tested frames, not an 81-frame production evolution. The different late-time
values demonstrate why future comparisons must use stored physical time, not
assume matching file indices imply simultaneous fields.

## Step 6B: full evolution and saved-curve comparison

Run the complete study, or split extraction and reporting:

```bash
PYENV_VERSION=research312 python scripts/31_compare_h_refinement_leading_edges.py
# Equivalent separate phases; analysis reads only saved curves:
PYENV_VERSION=research312 python scripts/31_compare_h_refinement_leading_edges.py --phase extract
PYENV_VERSION=research312 python scripts/31_compare_h_refinement_leading_edges.py --phase analyze
```

The default output is `/data/Nek5000_data/results/h_refinement/leading_edge/step6b/`. Use one of
these routes, not all commands consecutively: extraction refuses existing raw
case directories and reporting refuses an existing comparison directory.
`--phase extract --case N7_H` restricts extraction to a configured case; repeat
`--case` if needed. A failed extraction keeps per-frame checkpoints and an
`in_progress` manifest. Analysis requires complete manifests for all cases.

The producer builds one existing spectral plan per case, checks its complete
physical x/y grid against the Step 6A definition, and checks all element
coordinate signatures when applying it to every frame. Each concentration
plane is evaluated once, passed to both existing extractors, and released.
No interpolation, inverse-map, ownership, threshold, Moore tracing, or legacy
N7 numerical kernel is reimplemented or changed. No smoothing is introduced.

`raw/CASE/` preserves original unaligned results:

- `raw_curves.npz`: original physical times, indices, x/y grids, both methods'
  front curves, success masks and crossing counts.
- `frame_statistics.csv`: method-specific success/failure counts, mean,
  population standard deviation, min/max, peak-to-peak amplitude, and sampled
  plane finite/NaN diagnostics for every frame.
- `metadata.json`: physical definitions, source files and header times,
  source sizes/mtimes, geometry counts, plan diagnostics and completion status.
- `frames/fNNNNN.npz`: small per-frame checkpoints containing both methods.

`comparison/` contains separate derived data:

- `same_index_time_offsets.csv`: actual times for matching indices, signed
  offsets from VVH and the maximum cross-case spread at each shared index.
- `temporal_interpolation.csv`: each target's bracketing file indices, stored
  times, interpolation weight and valid counts.
- `aligned_fronts.npz`: aligned primary curves, per-case masks and the
  all-case common valid mask.
- `difference_metrics.csv`: H/VH minus the finest available numerical
  reference VVH, at every comparison time. VVH is not an exact solution or an
  independent convergence point.
- `aligned_morphology.csv` and `raw_frame_statistics.csv`: aligned common-mask
  morphology and original per-method morphology, respectively.
- `method_difference_history.csv` and `method_difference_points.csv`: raw
  same-frame extraction-method sensitivity; the latter lists every y position
  with a different front value or success mask, even if differences are tiny.
- `summary.json`: definitions, diagnostics, raw-source SHA256 hashes, summary
  values and confirmation that raw artifacts were preserved.
- PNG/PDF figures: full and selected aligned-time evolution overlays, mean-front,
  total RMS, shape RMS, bulk displacement, standard deviation, amplitude,
  and rightmost-versus-Moore histories.

### Time and mask definitions

The common timeline is **VVH's stored physical times inside the intersection
of all case time ranges**. Other cases use piecewise linear interpolation of
each `x_front(y,t)` between adjacent stored frames. Exact stored times are
copied using only that frame's validity. Interior targets require finite,
successful crossings in **both** neighboring frames. A failed crossing is
never filled or bridged. Targets outside a case's physical time range are
rejected; no temporal extrapolation is permitted.

At each aligned time, all H/VH/VVH metrics use the intersection of their valid
y masks. Coverage is reported. Each case and reference mean is computed on
that same support. The signed bulk difference is `mean(case)-mean(VVH)`;
positive values mean the case front is further downstream. Total metrics use
`case-VVH`. Shape metrics use
`(case-mean(case))-(VVH-mean(VVH))`. Therefore, on the common mask,
`total_RMS² = bulk_difference² + shape_RMS²`, up to floating-point roundoff.
All spanwise samples have equal weight; the periodic endpoint is excluded.

Empty support yields NaN metrics (JSON null), never zero differences. Raw
statistics describe each curve's own successful finite support; aligned
morphology describes the common support. Partial-support statistics must not
be read as complete-span morphology. Rightmost/Moore comparisons use raw
same-frame curves without time interpolation; identical-frame counts require
both matching front values (including matching NaNs) and matching masks.
Crossing-count differences are reported separately because the methods have
different candidate-selection semantics.

Linear time interpolation introduces a temporal approximation and can change
shape statistics between snapshots. Mean removal removes streamwise bulk
translation only; shape differences remain sensitive to spanwise phase and
structural changes. The fixed sampling grid is not a sampling-convergence
study. The anisotropic graded simulation meshes do not justify a global
observed h-order. Smaller reference differences alone are not evidence of
asymptotic convergence.

### Completed real-data validation

All three cases have 81 original frames at strictly increasing stored times
from 0 to 20, indexed 1 through 81. Each method finds 308/308 crossings in
every frame (24,948/24,948 per case and method), with no failed crossings or
sampled-plane NaNs. The target coordinates are exactly the Step 6A x/y arrays;
all 308,000 plane samples are mapped per case. The original raw histories and
per-frame checkpoints remain in `raw/CASE/` and are hashed in the comparison
summary. Previously saved h-refinement artifacts were unchanged.

Matching indices are not exactly simultaneous: the largest three-case time
spread is 0.004882859535 at index 7. The comparison uses all 81 stored VVH
times in the common range [0, 20]. H and VH curves are linearly interpolated
in time without extrapolation or failed-crossing filling. The all-case common
valid mask contains 308/308 spanwise points at every target time.

The following values are differences from **VVH as the finest available
numerical reference**, not errors against an exact solution. RMS quantities
are spanwise x-position differences; the signed bulk column is
`mean(case)-mean(VVH)`.

| VVH target time | Case | Total RMS | Signed bulk | Shape-only RMS |
| ---: | --- | ---: | ---: | ---: |
| 5.00282 | H | 0.02061 | -0.00486 | 0.02002 |
| 5.00282 | VH | 0.02129 | +0.00709 | 0.02007 |
| 10.00153 | H | 0.02823 | +0.00708 | 0.02733 |
| 10.00153 | VH | 0.06374 | +0.05008 | 0.03944 |
| 15.00154 | H | 0.06637 | -0.03610 | 0.05570 |
| 15.00154 | VH | 0.09574 | +0.04964 | 0.08186 |
| 20.00000 | H | 0.07532 | -0.02971 | 0.06921 |
| 20.00000 | VH | 0.08920 | +0.04526 | 0.07687 |

At time 20, the aligned mean fronts are H 16.05101, VH 16.12599 and VVH
16.08072. Their spanwise standard deviations are 0.03984, 0.04045 and
0.06587; peak-to-peak amplitudes are 0.18381, 0.16644 and 0.25997,
respectively. The complete 81-time morphology and metric histories are in the
CSV tables. VH has a smaller total RMS than H at only 6 of 81 target times,
and a smaller shape-only RMS at 8 of 81. These comparisons do not support a
monotonic refinement claim. Late-time differences can reflect bulk travel as
well as spanwise phase and morphology differences; this is a diagnostic
interpretation, not a demonstrated cause.

Rightmost crossing and Moore give exactly identical front positions and
success masks in all 81 frames of every case: per-frame RMS and maximum
differences are zero, and there are no differing y positions. Their candidate
crossing counts differ at 79 H, 93 VH and 48 VVH y/frame locations because
the two methods count candidates differently; these count differences do not
change the extracted fronts.
