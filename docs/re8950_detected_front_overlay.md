# Re8950 Detected-Front Overlay

The `GC8950_N7` concentration fields are processed by the spectral-element automatic front workflow. This comparison uses its `GC8950_N7_detected_front_timeseries.csv` output as the official simulation-front source. It does not read or use `front_simple.dat`.

Only rows with the established successful automatic tracking statuses, `selected_initial` and `selected_tracked`, enter the comparison. The earliest retained point defines `x0`, and the simulation coordinate is:

```text
x_front - x0 = x_front_auto(t) - x_front_auto(t_first)
```

This subtraction preserves direction: it does not apply absolute values, shift time, scale either axis, or calibrate an offset against the paper curve.

The digitized Cantero Figure 5a 3D Re8950 data come from the configured `paths.cantero_fig5a_re8950_csv`. Paper points are retained only within the automatic-front time range. Linear temporal interpolation evaluates the automatic displacement at those paper sample times; the workflow does not extrapolate. This temporal comparison is separate from the spectral-element spatial interpolation that produced the concentration fields.

The CSV outputs are:

- `GC8950_N7_Re8950_fig5a_comparison.csv`
- `GC8950_N7_Re8950_fig5a_summary.csv`

The comparison table reports the signed difference as the automatic spectral-front displacement minus the digitized Cantero Re8950 value, along with absolute, relative, and logarithmic differences. The digitized curve is treated as an external published comparison, not exact truth.

When plots are enabled, the outputs are:

- `GC8950_N7_Re8950_fig5a_overlay_linear.png`
- `GC8950_N7_Re8950_fig5a_overlay_loglog.png`
- `GC8950_N7_Re8950_fig5a_difference.png`
- `GC8950_N7_Re8950_fig5a_slumping_overlay.png`

Run the default comparison with:

```bash
python scripts/17_fig5a_detected_front_overlay.py --overwrite
```

Use `--no-plots` to write only the comparison and summary CSV files.
