# N7 Reconstructed x-t Four-Way Overlay

This workflow reconstructs N7 front trajectories independently from two automatic spectral-front inputs:

- Re3450: `front_detection_dir/N7_spectral/N7_detected_front_timeseries.csv`
- Re8950: `front_detection_dir/GC8950_N7_spectral/GC8950_N7_detected_front_timeseries.csv`

Only successful automatic detections from those CSV files are used. The workflow does not read or use `front_simple.dat`.

For each Reynolds number, the automatic positions are passed to the established front-kinematics implementation. It differentiates `x_front_auto(t)` with `numpy.gradient`, smooths the resulting velocity using the selected moving-average or Savitzky-Golay configuration, and reconstructs position by trapezoidal integration from the first automatic front position. The position samples themselves are not smoothed.

The two relative coordinates are defined independently:

```text
x_detected_relative = x_front_auto - x_front_auto[0]
x_reconstructed_relative = x_reconstructed - x_reconstructed[0]
```

Time is not shifted or resampled. No endpoint correction, fitted offset, absolute-value transformation, drift normalization, or paper calibration is applied.

The main figures include separate Re3450 and Re8950 pair overlays plus a
four-way overlay containing exactly:

1. Nek5000 Re3450 N7 reconstructed front
2. Nek5000 Re8950 N7 reconstructed front
3. Cantero Figure 5a 3D Re3450 digitized data
4. Cantero Figure 5a 3D Re8950 digitized data

Simulation reconstructions are shown as thin continuous lines without
markers. Paper data are shown as 2-point circle markers without connecting
lines and are drawn above the simulation curves so coincident Re3450 points
remain visible. Both paper artists contain the finite CSV points themselves;
the workflow rejects an empty finite paper dataset instead of creating a
legend-only entry.

The Re3450 reconstruction is compared only with the configured Re3450 paper dataset, and the Re8950 reconstruction only with the configured Re8950 paper dataset. Linear temporal interpolation evaluates reconstructed displacement at paper sample times inside each pair's overlapping time range. It is used only for the comparison tables, and it does not extrapolate.

The reconstructed timeseries outputs are:

- `Re3450_N7_detected_front_reconstructed_xt.csv`
- `Re8950_N7_detected_front_reconstructed_xt.csv`

The paired comparison and summary outputs are:

- `Re3450_N7_reconstructed_vs_Cantero_Re3450.csv`
- `Re8950_N7_reconstructed_vs_Cantero_Re8950.csv`
- `N7_Re3450_Re8950_reconstructed_xt_summary.csv`

The three figure outputs are:

- `Re3450_N7_reconstructed_vs_Cantero_Re3450.png`
- `Re8950_N7_reconstructed_vs_Cantero_Re8950.png`
- `N7_Re3450_Re8950_reconstructed_fourway_overlay.png`

Run the default workflow with:

```bash
python scripts/18_n7_reconstructed_xt_fourway_overlay.py --overwrite
```

To write only the five CSV outputs:

```bash
python scripts/18_n7_reconstructed_xt_fourway_overlay.py \
  --no-plots \
  --overwrite
```
