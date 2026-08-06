# Nek5000 Polynomial-Order Comparison

This repository compares Nek5000 simulation results computed with polynomial orders `N=5`, `N=7`, `N=9`, and `N=11`.

This repository now includes a working `t19p5` polynomial-order comparison workflow for concentration, velocity magnitude, and pressure fluctuation fields.

Raw simulation data is stored outside the repository under `/data/Nek5000_data`:

- `N5` (`GC3450_N5`): `/data/Nek5000_data/case_N5`
- `N7` (`GC3450_N7`): `/data/Nek5000_data/case_N7`
- `N9` (`GC3450_N9`): `/data/Nek5000_data/case_N9`
- `N11` (`GC3450_N11`): `/data/Nek5000_data/case_N11`
- `GC8950_N7`: `/data/Nek5000_data/GC8950_N7`

Configured digitized paper datasets are:

- Re3450: `/data/Nek5000_data/cantero/cantero_fig5a_3D_Re3450.csv`
- Re8950: `/data/Nek5000_data/cantero/cantero_fig5a_3D_Re8950.csv`

Intermediate processed data is stored under:

- `/data/Nek5000_data/postproc/poly_order_compare`

Final figures, tables, and reports are stored under:

- `/data/Nek5000_data/results/poly_order_compare`

## Current Completed Workflow

The current workflow compares `N5`, `N7`, and `N9` against `N11`, which is the reference case. The main comparison set is `t19p5`, using time-aligned files rather than identical file indices:

- `N5`: `f00079`
- `N7`: `f00079`
- `N9`: `f00079`
- `N11`: `f00040`

Slice extraction uses the nearest midspan `y` plane by default. For the current data this selects `y = 0.75`. The previous finite-thickness slab mode remains available with `--slice-mode slab`.

Supported comparison fields:

- `concentration`
- `velocity`
- `pressure`

Duplicate projected `(x,z)` points are averaged before interpolation onto the common grid.

## Project Layout

- `config/` contains YAML configuration files for paths and cases.
- `scripts/` contains entry-point scripts for each workflow stage.
- `src/nek_post/` contains the import-safe Python package.
- `tests/` contains minimal unit tests for the metrics helpers.

See [docs/script_index.md](docs/script_index.md) for a concise index of available scripts.

## Setup

This project uses a `src/` layout. Install it once in editable mode so scripts and tests can import `nek_post`:

```bash
PYENV_VERSION=research312 python -m pip install -e . --no-deps --no-build-isolation
```

Editable installation reflects source-code changes immediately without reinstalling. Existing commands remain unchanged, for example `python scripts/03_compare_poly_orders.py --help`.

## Check Expected Files

Run from the repository root:

```bash
python scripts/00_check_files.py
```

The script checks configured file paths only and writes the same report to `/data/Nek5000_data/postproc/poly_order_compare/logs/check_files.log`.

## Probe One Nek5000 File

Run the second utility from the repository root:

```bash
python scripts/01_probe_nek_file.py --case N11 --index 40
```

The script reads exactly one Nek5000 file with `pymech` and writes the same report to `/data/Nek5000_data/postproc/poly_order_compare/logs/probe_nek_file.log`.

## Run t19p5 Pipeline

Run the full configured time-aligned comparison pipeline:

```bash
python scripts/05_run_t19p5_pipeline.py --fields concentration,velocity,pressure --overwrite
```

The pipeline calls slice extraction, comparison, and plotting scripts for the `t19p5` comparison set.

## Run Multi-Time Pipeline

Run all configured multi-time comparison sets:

```bash
python scripts/07_run_multitime_pipeline.py \
  --comparison-sets t05,t10,t15,t19p5 \
  --fields concentration,velocity,pressure \
  --overwrite
```

Preview the commands without executing them:

```bash
python scripts/07_run_multitime_pipeline.py \
  --comparison-sets t05,t10,t15,t19p5 \
  --fields concentration,velocity,pressure \
  --dry-run
```

The multi-time runner only executes the existing slice extraction, comparison, and plotting scripts. It does not collect cross-time summary tables or create error-vs-time plots; cross-time summary collection will be handled by a later task.

## Collect Multi-Time Error Summary

Collect existing per-time error CSV files into one summary table:

```bash
python scripts/08_collect_multitime_error_summary.py \
  --comparison-sets t05,t10,t15,t19p5 \
  --fields concentration,velocity,pressure
```

The collector writes `/data/Nek5000_data/results/poly_order_compare/tables/multitime_error_summary.csv`. It does not recompute errors; the summary CSV will be used by later plotting tasks.

Plot error-versus-time figures from the collected summary:

```bash
python scripts/08_collect_multitime_error_summary.py \
  --comparison-sets t05,t10,t15,t19p5 \
  --fields concentration,velocity,pressure

python scripts/09_plot_multitime_error_summary.py
```

The plotter reads `multitime_error_summary.csv` and writes figures under `/data/Nek5000_data/results/poly_order_compare/figures/error_summary/`. It does not recompute errors.

## Plot Selected-Time Overlays

Create qualitative overlays from already interpolated comparison outputs:

```bash
python scripts/10_plot_selected_overlays.py \
  --comparison-sets t05,t10,t15,t19p5 \
  --fields concentration,velocity,pressure \
  --profile-z 0.5 \
  --profile-x 0.0 \
  --concentration-thresholds 0.01
```

The overlay script reads existing interpolated `.npz` files and writes figures under `/data/Nek5000_data/results/poly_order_compare/figures/overlays/`. It does not recompute errors and is mainly for qualitative comparison of `N5`, `N7`, `N9`, and `N11` at selected times.

## Run Individual Comparisons

```bash
python scripts/03_compare_poly_orders.py --comparison-set t19p5 --field concentration --overwrite
python scripts/03_compare_poly_orders.py --comparison-set t19p5 --field velocity --overwrite
python scripts/03_compare_poly_orders.py --comparison-set t19p5 --field pressure --overwrite
```

## Run Individual Plotting

```bash
python scripts/04_plot_summary.py --comparison-set t19p5 --field concentration
python scripts/04_plot_summary.py --comparison-set t19p5 --field velocity
python scripts/04_plot_summary.py --comparison-set t19p5 --field pressure
```

## N7 Leading-Edge Evolution

Generate the Re=3450 N7 Figure-4-style spanwise leading-edge evolution from the
fixed physical plane `z=0.04`, concentration contour `C=0.1`, and production
target-time spacing `delta_t=0.25`. The right-moving current is isolated before
extraction with the strict physical domain `x > 0.0`:

```bash
python scripts/19_compute_leading_edge_evolution.py \
  --case N7 \
  --x-min 0.0 \
  --workers 2 \
  --overwrite
python scripts/20_plot_leading_edge_evolution.py --case N7 --overwrite
```

Run the implemented Moore comparison into a distinct artifact directory so it
does not replace the production-default result:

```bash
python scripts/19_compute_leading_edge_evolution.py \
  --case N7 \
  --extraction-method moore-boundary \
  --x-min 0.0 \
  --output-dir /path/to/leading_edge/N7/moore_boundary \
  --overwrite
```

The workflow uses element-aware GLL interpolation to a uniform periodic y grid
with `dense_ny = 2 * native_ny`; it does not apply an FFT directly to raw
element-local arrays or add DNS resolution. Recompute CSV artifacts only when
the data or numerical parameters change. Plot-only reruns read those CSVs and
never access `.fNNNNN` files, making them appropriate for figure-formatting
changes.

The compute command supports `--extraction-method rightmost-crossing` and
`--extraction-method moore-boundary`; `rightmost-crossing` remains the
production default. Moore boundary tracing is a pure-Python, Fortran-derived
Moore-neighbour method that preserves the supplied neighbour order and
indicator turn rule while using deterministic start selection and
directed-edge closure. It traces finite low-side `C <= threshold` reconstructed
grid nodes adjacent to the `C > threshold` region; exact-threshold nodes are on
the low side. The y direction is periodic, x is non-periodic, and the result
uses grid-node locations rather than sub-grid threshold interpolation. No
Fortran is compiled or called, and GridEnhancer, FFT, and FFTW are not used.
Both methods consume the same spectral horizontal field: x remains fixed at
configured `nx` with no extraction-stage upsampling, and y remains
`dense_ny = y_upsample_factor * native_ny` with production factor `2`.
`marching-squares-ad` remains planned and unavailable.

The common dispatcher removes `x <= x_min` columns before either method runs;
it does not trace the full domain and filter afterward, and it does not
interpolate a value at `x=0`. Production metadata records
`extraction_x_min=0.0` and
`extraction_x_condition=strict-greater-than`. Plot-only reruns use that metadata
to keep the displayed x range within the positive half-domain. If several
full-span, one-winding Moore traces exist there, selection prefers greatest y
coverage, then greatest median and mean rowwise physical x, followed by stable
component/start/indicator tie-breaks.

Because legacy artifact filenames do not contain the method name, use a
method-specific `--output-dir` such as
`.../leading_edge/N7/moore_boundary` when preserving both outputs. The
plot-only command reads either method's common CSV artifacts without accessing
Nek files and retains the same appearance.

The compute command's configured production default is two processes. The first
frame remains serial and defines the reusable spectral interpolation plan; only
later frame descriptors enter the process pool, and workers return reduced
one-dimensional leading-edge results rather than full concentration planes.
Use `--workers 1` as the deterministic serial baseline. More workers can add
memory and I/O pressure. The completed full-N7 benchmark retained workers=2 as
the production default; workers=4 remains an optional manual override. The
plot-only command is independent of workers and raw Nek files.

Benchmark the existing compute workflow in fresh GNU-timed processes, with
workers=1 as the mandatory exact-output baseline:

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

The benchmark stores isolated run artifacts, logs, GNU-time metrics, and stable
run/summary CSV reports outside the production leading-edge directory. It
requires exact scientific equivalence and does not alter the configured
workers=2 default. Run it on an otherwise idle machine; see the detailed guide
for the full-resolution command and measurement limitations.

For the completed full-N7 benchmark using all discovered frames, `nx=1000`, and
three repetitions, median wall times were 516.30 s for workers=1, 367.69 s for
workers=2, and 348.64 s for workers=4. These correspond to speedups of 1.00000,
1.40417, and 1.48090, respectively, and every configuration produced
scientifically equivalent artifacts. Workers=2 reduced wall time by about 28.8%
relative to workers=1; workers=4 was only about 5.2% faster than workers=2. GNU
time's maximum RSS is not the aggregate memory usage of the complete parent and
worker process tree.

See
[docs/leading_edge_evolution.md](docs/leading_edge_evolution.md) for the full
physical definition, CLI flags, outputs, and validation commands.

## Output Locations

Postprocessed slices and interpolated files are written under:

- `/data/Nek5000_data/postproc/poly_order_compare`

Final tables and figures are written under:

- `/data/Nek5000_data/results/poly_order_compare`
