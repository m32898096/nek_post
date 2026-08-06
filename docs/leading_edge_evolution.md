# N7 Leading-Edge Evolution

## Physical objective

This workflow constructs a Figure-4-style view of how the gravity-current
leading edge varies across the periodic span. At each selected simulation time,
the selected extraction method produces a common `x_front(y)` representation
of the `C = 0.1` boundary on the horizontal plane `z = 0.04`. The production
default uses the rightmost physical-x threshold intersection independently for
every sampled physical y coordinate.

The production definition is:

- case: `N7`
- Reynolds number: `3450`
- horizontal plane: `z = 0.04`
- concentration contour: `C = 0.1`
- extraction method: `rightmost-crossing`
- extraction domain: strict `x > 0.0`
- target-time spacing: `delta_t = 0.25`
- spanwise upsampling factor: `2`
- compute workers: `2`

These defaults are stored in the dedicated `leading_edge` section of
`config/cases.yaml`.

The reference paper used `Delta t = 0.28`, while this project uses
`Delta t = 0.25` to match the natural cadence of the N7 simulation outputs.

## Extraction methods

Two extraction definitions are implemented for comparison:

Before either method runs, the common dispatcher restricts the reconstructed
field to physical columns satisfying strict `x > x_min`. The N7 production
configuration records `x_min = 0.0`; columns at `x = 0` and all negative-x
columns are removed before threshold intersections, boundary-candidate
construction, connected-component labeling, or tracing. No value is
interpolated at the bound. This isolates the right-moving gravity current and
gives every extraction method the same scientific domain.

- `rightmost-crossing` is the production default. For each supplied y row, it
  finds the existing exact-threshold and strict sign-change intersections along
  x and retains the rightmost sub-grid intersection. Its adapter delegates to
  the validated legacy extractor, so plateau, NaN, and crossing behavior are
  unchanged.
- `moore-boundary` is a pure-Python, Fortran-derived Moore-neighbour tracing
  method. It traces finite heavy-side `C > threshold` reconstructed grid nodes
  that have at least one light-side `C <= threshold` eight-neighbour.
  Exact-threshold values are light-side; non-finite values belong to neither
  side. The y index wraps periodically and x is non-periodic. The selected
  trace constrains which downstream heavy-to-light row intersections are
  retained. Their `x_front(y)` coordinates use the same sub-grid physical-x
  linear interpolation and rightmost exact-plateau representation as
  `rightmost-crossing`.

The Moore implementation retains the supplied Fortran one-based neighbour
order, initial indicator `4`, and the wrapped `+5` turn before each local
search. It is implemented entirely in Python and NumPy: no Fortran is compiled
or called, and neither GridEnhancer, Fourier zero-padding, FFT, nor FFTW is
used. The Python method intentionally replaces the Fortran fixed start point
with a deterministic front-biased, upper-seam-preferred start and replaces
right-x-edge termination with directed-edge closure. Candidate construction
uses the predecessor boundary definition: a heavy-fluid grid point adjacent to
a light-fluid grid point. It is therefore an alternative extraction definition,
not a claim of bitwise or complete-program equivalence or greater accuracy.

When that candidate mask contains disconnected boundaries, the Python method
labels periodic-y, non-periodic-x eight-connected components. Full-span
one-winding traces are compared first by greatest unique-y count, then greatest
median rowwise physical x, then greatest mean rowwise physical x, followed by
deterministic component, start, and indicator tie-breaks. Component pixel count
does not outrank streamwise position in that choice. If no full-span winding
trace exists, the established primary-component fallback remains available for
partial-y data. This connected-component selection is a Python robustness
adaptation motivated by full N7 data, not behavior inherited from the supplied
Fortran.

The selected component can still contain branches and small contractible local
cycles. The production method therefore evaluates deterministic Moore walks
from every boundary pixel on both sides of the periodic seam and all eight
initial indicators. Each closed walk is assigned an integer periodic-y winding
number. When the primary component spans every y row, the selected trace must
wind exactly once (`abs(winding_number) == 1`); zero-winding local cycles are
rejected even when they are farther forward in x. For a component that does not
span the full y domain, the best deterministic zero-winding closed trace remains
an allowed fallback. This topology-aware trace selection is another Python
robustness adaptation and is not claimed to originate in the supplied Fortran
or to provide greater physical accuracy.

Both methods consume exactly the same element-aware spectral horizontal field.
The physical x grid retains the configured fixed `nx`, with no extraction-stage
upsampling. The endpoint-excluded periodic y grid retains
`dense_ny = y_upsample_factor * native_ny`, and the production y upsampling
factor remains `2`. `marching-squares-ad` remains planned and is not
implemented.

## Numerical pipeline

The complete processing sequence is:

```text
raw Nek5000 element-local GLL data
  -> element-aware spectral interpolation at fixed physical z
  -> uniform periodic physical-y target grid
  -> dense_ny = 2 * native_ny
  -> strict physical extraction domain x > 0.0
  -> selected leading-edge extraction method at C = 0.1
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
  physical parameters, strict extraction-x domain, spanwise resolution,
  endpoint policy, algorithm version, and inverse-mapping diagnostics.
- `N7_leading_edge_evolution.png`: raster Figure-4-style overlay.
- `N7_leading_edge_evolution.pdf`: the same figure in vector form.

The timeseries schema is:

```text
case,file_index,source_file,target_time,actual_time,time_error,y,x_front,success,crossing_count,threshold,z_target,nx,native_ny,dense_ny,y_upsample_factor
```

The metadata schema is:

```text
case,extraction_method,extraction_x_min,extraction_x_condition,n_input_frames,n_selected_frames,actual_time_start,actual_time_end,target_time_spacing,threshold,z_target,nx,native_ny,dense_ny,y_upsample_factor,y_min,y_max_periodic_endpoint,periodic_endpoint_included,algorithm_version,inverse_mapping_target_count,inverse_mapping_success_count,inverse_mapping_failure_count,ambiguous_boundary_point_count,maximum_successful_residual,maximum_iteration_count
```

NaN leading-edge values are preserved as `nan`. They indicate y rows without a
finite threshold crossing and remain gaps in the figure.

For `rightmost-crossing`, `crossing_count` is the number of threshold
intersections in a y row. For `moore-boundary`, it is the number of downstream
threshold intersections in that row whose heavy-side supporting node belongs
to the selected Moore trace.

## Two-step command-line workflow

The recommended production workflow is:

```bash
python scripts/19_compute_leading_edge_evolution.py \
  --case N7 \
  --x-min 0.0 \
  --workers 2 \
  --overwrite
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
| `--extraction-method` | Extraction engine: `rightmost-crossing` or `moore-boundary`. |
| `--x-min` | Strict extraction lower bound; production retains `x > 0.0`. |
| `--workers` | Process count for later frames; configured default `2`. |
| `--contour-time-spacing` | Regular target-time spacing; default `0.25`. |
| `--all-frames` | Select all processed snapshots instead of spaced targets. |
| `--output-dir` | Override the dynamic CSV artifact directory. |
| `--overwrite` | Permit replacement of both CSV artifacts. |

`--all-frames` and an explicitly supplied `--contour-time-spacing` are mutually
exclusive. Both CSV paths are preflighted before any Nek snapshot is read or
processed.

The first frame is always read and reduced serially because its stationary
geometry defines the one reusable spectral horizontal interpolation plan. With
`--workers 2`, only subsequent frame descriptors are submitted to the process
pool. The plan, x and y coordinates, threshold, and extraction-x bound are
installed once in each worker by the process initializer; they are not
submitted with every task and are not rebuilt in workers. Each worker reads its
own snapshot and returns only the reduced one-dimensional leading-edge arrays
and scalar diagnostics. Raw Nek objects and full two-dimensional concentration
planes are discarded in the worker.

`--workers 1` is the deterministic serial baseline and is also the mode to use
with the package API's custom frame-reader injection. The completed full-N7
benchmark supports retaining two workers as the configured production default;
four workers remain available as a manual override. Increasing the process
count can still raise memory use and I/O pressure.

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

The legacy artifact filenames do not include the extraction method. To retain
both definitions, give the compute command a method-specific directory, for
example:

```bash
python scripts/19_compute_leading_edge_evolution.py \
  --case N7 \
  --extraction-method moore-boundary \
  --x-min 0.0 \
  --output-dir /path/to/leading_edge/N7/moore_boundary \
  --overwrite
```

The plot-only command reads the recorded `extraction_x_min` and prevents the
x-axis range from extending below that bound. It can read the resulting common
CSV artifacts without accessing Nek files, and its line appearance is
independent of the extraction method.

Recompute with script 19 only when the source snapshots, frame range, physical
definition, target grid, upsampling, threshold, or time selection changes.
Figure-formatting changes require only script 20; these reruns do not read any
`.fNNNNN` file or repeat spectral interpolation. Plot-only execution is
independent of the compute worker count and does not expose a worker option.

## Worker-count benchmark

Script 21 measures the existing script-19 compute command in fresh Python
processes for `workers=1`, `workers=2`, and `workers=4`. Each run includes plan
construction, Nek snapshot reads, process-pool startup where applicable,
spectral interpolation, leading-edge extraction, time selection, and CSV
writing. On Linux, `/usr/bin/time` supplies elapsed wall time, user CPU time,
system CPU time, and maximum resident set size in KiB through a dedicated
machine-readable metrics file.

Use a bounded frame range and reduced x resolution for an initial benchmark:

```bash
PYENV_VERSION=research312 python \
  scripts/21_benchmark_leading_edge_workers.py \
  --case N7 \
  --start-index 37 \
  --end-index 52 \
  --nx 500 \
  --worker-counts 1,2,4 \
  --repeats 2 \
  --all-frames \
  --overwrite
```

Run the full production-resolution comparison separately:

```bash
PYENV_VERSION=research312 python \
  scripts/21_benchmark_leading_edge_workers.py \
  --case N7 \
  --nx 1000 \
  --worker-counts 1,2,4 \
  --repeats 3 \
  --overwrite
```

### Completed full-N7 results

The full benchmark used all discovered N7 frames, `nx=1000`, `z_target=0.04`,
`threshold=0.1`, `y_upsample_factor=2`, `contour_time_spacing=0.25`, and three
measured repetitions per worker count. The median results were:

| Workers | Median wall time (s) | Speedup vs. workers=1 | Scientifically equivalent |
| ---: | ---: | ---: | :---: |
| 1 | 516.30 | 1.00000 | True |
| 2 | 367.69 | 1.40417 | True |
| 4 | 348.64 | 1.48090 | True |

Workers=2 reduced median wall time by approximately 28.8% relative to the
serial baseline. Workers=4 was approximately 5.2% faster than workers=2, a much
smaller incremental gain. All three configurations produced scientifically
equivalent CSV artifacts. On this evidence, workers=2 remains the configured
production default, while workers=4 is an optional manual override when the
additional process and I/O load is acceptable.

GNU time's maximum-RSS measurement must not be interpreted as the aggregate
memory simultaneously used by the complete process tree. In particular, it is
not a total of the parent and all worker resident sets.

Run benchmarks while the machine is otherwise idle, using the same data
location and scientific parameters for every worker count. Wall time determines
practical speed. User CPU time can rise as more processes do work, and peak RSS
can rise because each worker reads snapshots and holds interpolation state.
One short run is not enough evidence for changing a default.

Every measured run has a distinct directory below
`paths.results_root/leading_edge_benchmarks/CASE/runs`, containing its two CSV
artifacts, `stdout.log`, `stderr.log`, and `metrics.txt`. Warmups use disposable
directories and do not enter the reports. The benchmark writes:

- `leading_edge_workers_benchmark.csv`, with one stable-schema row per measured
  run, including timing, RSS, hashes, paths, return status, and equivalence.
- `leading_edge_workers_summary.csv`, with worker-count medians, wall-time
  range, speedup relative to the workers=1 median, efficiency, median peak RSS,
  and aggregate equivalence.

Workers=1 is the mandatory numerical baseline. SHA-256 equality is checked
first; differing hashes trigger exact structured comparison with zero relative
and absolute tolerance and equal-NaN handling. Different source frames, times,
y values, `x_front` values or NaN locations, success/crossing arrays, physical
settings, endpoint policy, or metadata invalidate the benchmark, and no valid
speedup is claimed. The benchmark never writes into the production
`leading_edge/N7` directory and never changes `config/cases.yaml`; workers=2
remains the configured production default unless separately reviewed.

## Validation commands

Use one snapshot for a quick CSV-only smoke test:

```bash
python scripts/19_compute_leading_edge_evolution.py \
  --start-index 1 \
  --end-index 1 \
  --workers 1 \
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
  --workers 2 \
  --output-dir /tmp/n7-leading-edge-small \
  --overwrite

python scripts/20_plot_leading_edge_evolution.py \
  --output-dir /tmp/n7-leading-edge-small \
  --overwrite
```

Run the full configured N7 production workflow with:

```bash
python scripts/19_compute_leading_edge_evolution.py \
  --case N7 \
  --workers 2 \
  --overwrite
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
