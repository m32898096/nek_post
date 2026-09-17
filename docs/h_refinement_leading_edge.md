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
grid. Its default output tree is `results/h_refinement/leading_edge/METHOD/CASE`.
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
`results/h_refinement/leading_edge/step6a_smoke/`: `report.json` and twelve
small NPZ curve files, one per case, frame and method. They contain only the
tested frames, not an 81-frame production evolution. The different late-time
values demonstrate why future comparisons must use stored physical time, not
assume matching file indices imply simultaneous fields.
