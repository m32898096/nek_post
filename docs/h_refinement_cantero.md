# H-refinement Cantero mean-front comparison

Step 5 applies the established Re=3450 planar Cantero workflow independently
to `N7_H`, `N7_VH`, and `N7_VVH`. It does not use the midspan slice from Steps
2–4 and does not use the leading-edge detector. The numerical path is:

1. composite physical GLL quadrature of concentration in z;
2. composite physical GLL quadrature in y divided by the spanwise length;
3. first positive-x downward crossing of mean equivalent height at
   `delta=0.01`, starting from physical `reference_x=0.0`;
4. `numpy.gradient` on the actual stored physical times;
5. the established 11-point moving-average velocity smoothing;
6. trapezoidal velocity integration from the first successful absolute front;
7. comparison of reconstructed relative displacement with digitized Cantero
   Figure 5a, Re=3450.

The GLL integration, threshold detector, smoothing, reconstruction, and paper
comparison functions are shared with the p-refinement workflow. Script 26 now
accepts an arbitrary ordered compatible case set; its no-argument N5/N7/N9
configuration and artifacts are unchanged.

## Commands

The h-refinement artifacts were generated from the repository root with:

```bash
for case in N7_H N7_VH N7_VVH; do
  PYENV_VERSION=research312 python scripts/24_compute_cantero_mean_front.py \
    --case "$case" \
    --output-dir results/h_refinement/cantero_mean_front \
    --stationary-geometry-fast-path
done

PYENV_VERSION=research312 python scripts/26_overlay_cantero_re3450_multicase.py \
  --cases N7_H N7_VH N7_VVH \
  --front-root results/h_refinement/cantero_mean_front \
  --output-dir results/h_refinement/cantero_re3450 \
  --reference-case N7_VVH \
  --include-input-summary
```

The stationary-geometry option builds the normal plan from frame 1 and uses
the unchanged quadrature and assembly plan for later concentration-only reads.
It checks element count and scalar shape on every frame. It also checks the
last frame's complete coordinate signatures and element ordering against the
first frame before publishing. For the real run, independent first/last checks
passed for all three cases. A two-frame H run was byte-identical to the default
full-coordinate-validation path.

## Real-data extraction

All source series contain 81 strictly time-ordered frames from `t=0` to `t=20`.
Every frame has one retained threshold crossing; no frame failed. The grids
have different internal time values even at equal file indices. Relative to
VVH at the same indices, the largest time difference is .00488286 for H and
.00311818 for VH. Reconstruction uses each case's own stored time vector.

| Case | Successful / input | Failed | First x_F | Last x_F | Raw displacement | Reconstructed displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N7_H | 81 / 81 | 0 | 8.00785638 | 16.10495425 | 8.09709788 | 8.08985823 |
| N7_VH | 81 / 81 | 0 | 8.01707031 | 16.17271245 | 8.15564214 | 8.14027582 |
| N7_VVH | 81 / 81 | 0 | 8.00689202 | 16.14786939 | 8.14097736 | 8.12999017 |

The sampled fronts increase monotonically. The established second-order
one-sided gradient gives one small negative raw endpoint velocity for VH at
`t=0` (-.00599212); its smoothed velocity is positive. This is retained rather
than edited because the temporal kernel is unchanged.

## Paper and numerical-reference comparisons

The Cantero comparison linearly interpolates each reconstructed numerical
trajectory onto the 34 paper times in the overlapping interval
`0.56721 <= t <= 17.318`. Differences are numerical minus paper.

| Case | Paper MAE | Paper RMS | Paper max abs. | Reconstructed slumping velocity | Paper slumping velocity |
| --- | ---: | ---: | ---: | ---: | ---: |
| N7_H | .05661847 | .06095101 | .10420237 | .40974459 | .41078592 |
| N7_VH | .05102393 | .05615026 | .09912416 | .41527993 | .41078592 |
| N7_VVH | .04912990 | .05522939 | .10240600 | .40833707 | .41078592 |

For the finest-reference diagnostics, VVH is linearly interpolated only at
each case's actual times inside `[0,20]`; no extrapolation occurs. Interpolation
is used only for the comparison table and does not change extraction,
velocity, smoothing, or reconstruction. VVH is the finest available numerical
reference, not an exact solution.

| Case vs VVH | Front-position MAE / RMS / max | Raw-velocity MAE / RMS / max | Smoothed-velocity MAE / RMS / max | Reconstructed-front MAE / RMS / max |
| --- | --- | --- | --- | --- |
| N7_H | .01606175 / .02114357 / .04291513 | .00640020 / .00832545 / .02199569 | .00418773 / .00548569 / .01201597 | .01591321 / .02017348 / .04013194 |
| N7_VH | .03218575 / .03834870 / .06366003 | .00680859 / .00994779 / .04919834 | .00486424 / .00661266 / .01753620 | .02118899 / .02671877 / .04636582 |

These differences are not monotonic with element count, so no observed h-order
is calculated. The front is a threshold-derived integral diagnostic, and the
comparison includes spatial discretization, temporal sampling, smoothing,
and evolving-flow phase/morphology effects. The Cantero digitization is a
literature comparison rather than an exact solution.

## Artifacts

`results/h_refinement/cantero_mean_front/` contains one full Phase-2 CSV per
case. `results/h_refinement/cantero_re3450/` contains reconstructed timeseries,
per-case Cantero comparison tables, the combined summary, input success/failure
summary, time-aligned VVH comparison tables, and linear/log-log overlays. The
N5/N7/N9 canonical result tree is not read for writing or modified.
