# Nek5000 Polynomial-Order Comparison

This repository is a lightweight Python skeleton for comparing Nek5000 simulation results computed with polynomial orders `N=5`, `N=7`, `N=9`, and `N=11`.

Raw simulation data is stored outside the repository under `/data/Nek5000_data`:

- `/data/Nek5000_data/case_N5`
- `/data/Nek5000_data/case_N7`
- `/data/Nek5000_data/case_N9`
- `/data/Nek5000_data/case_N11`

Intermediate processed data is stored under:

- `/data/Nek5000_data/postproc/poly_order_compare`

Final figures, tables, and reports are stored under:

- `/data/Nek5000_data/results/poly_order_compare`

## Initial Workflow

1. Check that the expected input files exist.
2. Inspect one Nek5000 field file.
3. Extract a midspan `y`-slice.
4. Interpolate all polynomial orders onto a common `x-z` grid.
5. Compute errors against `N11`, which is the reference case.
6. Generate plots and summary tables.

## Project Layout

- `config/` contains YAML configuration files for paths and cases.
- `scripts/` contains entry-point scripts for each workflow stage.
- `src/nek_post/` contains the import-safe Python package skeleton.
- `tests/` contains minimal unit tests for the metrics helpers.

This repository currently includes placeholders only. The analysis workflow will be implemented later.

## Check Expected Files

Run the first utility from the repository root:

```bash
python scripts/00_check_files.py
```

The script checks configured file paths only and writes the same report to `/data/Nek5000_data/postproc/poly_order_compare/logs/check_files.log`.

## Probe One Nek5000 File

Run the second utility from the repository root:

```bash
python scripts/01_probe_nek_file.py
```

By default, it probes the configured reference case and last configured file index. Override those with `--case` and `--index`, for example:

```bash
python scripts/01_probe_nek_file.py --case N11 --index 80
```

The script reads exactly one Nek5000 file with `pymech` and writes the same report to `/data/Nek5000_data/postproc/poly_order_compare/logs/probe_nek_file.log`.
