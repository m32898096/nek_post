# N7 Leading-Edge Evolution

## Physical objective

This workflow constructs a Figure-4-style view of how the gravity-current
leading edge varies across the periodic span. At each selected simulation time,
the leading edge is the rightmost physical-x intersection of the concentration
contour with `C = 0.1`, evaluated independently for every sampled physical y
coordinate on the horizontal plane `z = 0.04`.

The production definition is:

- case: `N7`
- Reynolds number: `3450`
- horizontal plane: `z = 0.04`
- concentration contour: `C = 0.1`
- target-time spacing: `delta_t = 0.25`
- spanwise upsampling factor: `2`

These defaults are stored in the dedicated `leading_edge` section of
`config/cases.yaml`.

The reference paper used `Delta t = 0.28`, while this project uses
`Delta t = 0.25` to match the natural cadence of the N7 simulation outputs.

## Numerical pipeline

The complete processing sequence is:

```text
raw Nek5000 element-local GLL data
  -> element-aware spectral interpolation at fixed physical z
  -> uniform periodic physical-y target grid
  -> dense_ny = 2 * native_ny
  -> rightmost C = 0.1 crossing in every y row
  -> nearest-snapshot time selection
  -> tidy timeseries and metadata CSV artifacts
  -> validated CSV artifact reader
  -> Figure-4-style curve overlay
```

The raw data consists of spectral elements with element-local GLL nodes. These
arrays are not samples of one global, uniformly spaced Fourier grid: element
interfaces can duplicate physical coordinates, and GLL points are nonuniform
inside each element. Passing the raw arrays directly to an FFT would therefore
assign the wrong global sampling interpretation.

Instead, the workflow performs physical-to-reference inversion within candidate
elements and tensor-product GLL barycentric interpolation. This evaluates the
concentration at requested physical `(x, y, z)` points while retaining the
spectral-element geometry.

This differs from a global Fourier/Chebyshev representation, where a field is
already expressed on globally organized periodic and wall-normal bases and a
Fourier resampling operation may be meaningful. The present Nek5000 snapshots
must first be interpreted element by element.

The factor-of-two y operation only provides denser post-processing samples of
the existing spectral-element interpolant. It does not add physical modes,
recover unresolved scales, or increase the DNS resolution.

## Periodic endpoint policy

The computational target coordinates use a periodic interval with the upper
endpoint excluded. Thus `y_max` is not duplicated in the stored `y`, `x_front`,
or CSV rows. For plotting only, each curve is copied and closed with:

```text
y_plot = append(y, y_max_periodic_endpoint)
x_plot = append(x_front, x_front[0])
```

The computational arrays remain unchanged. If the first leading-edge value is
NaN, the plotted periodic seam remains discontinuous; it is not filled or
smoothed.

## Outputs

By default, outputs are written below:

```text
paths.results_root/leading_edge/CASE
```

For the production case this contains:

- `N7_leading_edge_timeseries.csv`: one row for every selected frame and
  computational y coordinate.
- `N7_leading_edge_metadata.csv`: one row describing the input/selection sizes,
  physical parameters, spanwise resolution, endpoint policy, algorithm version,
  and inverse-mapping diagnostics.
- `N7_leading_edge_evolution.png`: raster Figure-4-style overlay.
- `N7_leading_edge_evolution.pdf`: the same figure in vector form.

The timeseries schema is:

```text
case,file_index,source_file,target_time,actual_time,time_error,y,x_front,success,crossing_count,threshold,z_target,nx,native_ny,dense_ny,y_upsample_factor
```

The metadata schema is:

```text
case,n_input_frames,n_selected_frames,actual_time_start,actual_time_end,target_time_spacing,threshold,z_target,nx,native_ny,dense_ny,y_upsample_factor,y_min,y_max_periodic_endpoint,periodic_endpoint_included,algorithm_version,inverse_mapping_target_count,inverse_mapping_success_count,inverse_mapping_failure_count,ambiguous_boundary_point_count,maximum_successful_residual,maximum_iteration_count
```

NaN leading-edge values are preserved as `nan`. They indicate y rows without a
finite threshold crossing and remain gaps in the figure.

## Two-step command-line workflow

The recommended production workflow is:

```bash
python scripts/19_compute_leading_edge_evolution.py --case N7 --overwrite
python scripts/20_plot_leading_edge_evolution.py --case N7 --overwrite
```

Script 19 reads the configured `.fNNNNN` snapshots, performs the numerical
workflow, and writes only the timeseries and metadata CSV artifacts. Its flags
are:

| Compute flag | Meaning |
| --- | --- |
| `--case` | Configured case label; default `N7`. |
| `--file-prefix` | Exact prefix before `.fNNNNN`; default `GC0`. |
| `--start-index` | Optional inclusive first snapshot index. |
| `--end-index` | Optional inclusive final snapshot index. |
| `--nx` | Uniform physical-x target count; default `1000`. |
| `--z-target` | Fixed physical horizontal-plane coordinate; default `0.04`. |
| `--threshold` | Leading-edge concentration contour; default `0.1`. |
| `--y-upsample-factor` | Multiplier defining `dense_ny`; default `2`. |
| `--contour-time-spacing` | Regular target-time spacing; default `0.25`. |
| `--all-frames` | Select all processed snapshots instead of spaced targets. |
| `--output-dir` | Override the dynamic CSV artifact directory. |
| `--overwrite` | Permit replacement of both CSV artifacts. |

`--all-frames` and an explicitly supplied `--contour-time-spacing` are mutually
exclusive. Both CSV paths are preflighted before any Nek snapshot is read or
processed.

Script 20 reads only those existing CSV artifacts and writes PNG and PDF. It
does not discover, open, or interpolate Nek5000 files. Its flags are:

| Plot flag | Meaning |
| --- | --- |
| `--case` | Case label expected in the CSV artifacts; default `N7`. |
| `--timeseries-csv` | Override the existing timeseries CSV input. |
| `--metadata-csv` | Override the existing metadata CSV input. |
| `--output-dir` | Override the figure output directory. |
| `--reynolds-number` | Reynolds number used in the title; default `3450`. |
| `--overwrite` | Permit replacement of both figure artifacts. |

With no path overrides, both commands use
`paths.results_root/leading_edge/CASE`. The plot command preflights PNG and PDF
together before reading either CSV.

Recompute with script 19 only when the source snapshots, frame range, physical
definition, target grid, upsampling, threshold, or time selection changes.
Figure-formatting changes require only script 20; these reruns do not read any
`.fNNNNN` file or repeat spectral interpolation. Multiprocessing and worker
options are intentionally not included yet.

## Validation commands

Use one snapshot for a quick CSV-only smoke test:

```bash
python scripts/19_compute_leading_edge_evolution.py \
  --start-index 1 \
  --end-index 1 \
  --output-dir /tmp/n7-leading-edge-single \
  --overwrite
```

Use a small range to validate interpolation, selection, artifact reading, and
both figure formats:

```bash
python scripts/19_compute_leading_edge_evolution.py \
  --start-index 1 \
  --end-index 5 \
  --nx 300 \
  --output-dir /tmp/n7-leading-edge-small \
  --overwrite

python scripts/20_plot_leading_edge_evolution.py \
  --output-dir /tmp/n7-leading-edge-small \
  --overwrite
```

Run the full configured N7 production workflow with:

```bash
python scripts/19_compute_leading_edge_evolution.py --case N7 --overwrite
python scripts/20_plot_leading_edge_evolution.py --case N7 --overwrite
```

Inspect the metadata and verify the target-grid relationship with:

```bash
python -c 'import csv; p="/data/Nek5000_data/results/poly_order_compare/leading_edge/N7/N7_leading_edge_metadata.csv"; r=next(csv.DictReader(open(p))); print(r); assert int(r["dense_ny"]) == 2 * int(r["native_ny"])'
```

The compute command also prints `native_ny`, `dense_ny`,
`y_upsample_factor`, and the explicit equality check in its success summary.

## Interpretation limits

The workflow does not smooth leading-edge curves, interpolate missing y rows,
classify merging or splitting events, or infer cleft trajectories. Those are
separate physical interpretation tasks and are not encoded in these artifacts.
