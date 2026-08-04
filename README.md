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
fixed physical plane `z=0.04` and concentration contour `C=0.1`:

```bash
python scripts/19_plot_leading_edge_evolution.py --overwrite
```

The workflow uses element-aware GLL interpolation to a uniform periodic y grid
with `dense_ny = 2 * native_ny`; it does not apply an FFT directly to raw
element-local arrays or add DNS resolution. See
[docs/leading_edge_evolution.md](docs/leading_edge_evolution.md) for the full
physical definition, CLI flags, outputs, and validation commands.

## Output Locations

Postprocessed slices and interpolated files are written under:

- `/data/Nek5000_data/postproc/poly_order_compare`

Final tables and figures are written under:

- `/data/Nek5000_data/results/poly_order_compare`
