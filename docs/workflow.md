# Workflow

This repository contains a working Nek5000 polynomial-order comparison workflow for `N5`, `N7`, `N9`, and `N11`. The current completed workflow compares `N5`, `N7`, and `N9` against `N11` for concentration, velocity magnitude, and pressure fluctuation fields.

## Data Layout

Raw Nek5000 data lives outside the repository:

- `/data/Nek5000_data/case_N5`
- `/data/Nek5000_data/case_N7`
- `/data/Nek5000_data/case_N9`
- `/data/Nek5000_data/case_N11`

Generated post-processing outputs are written under:

- `/data/Nek5000_data/postproc/poly_order_compare`

Final CSV tables and PNG figures are written under:

- `/data/Nek5000_data/results/poly_order_compare`

Comparison and plotting scripts read existing generated slice or interpolated `.npz` files. They do not read raw Nek5000 `.f` files.

## t19p5 Comparison Set

The primary comparison set is `t19p5`, configured near physical time `t = 19.5`. File indices are time-aligned across polynomial orders rather than identical:

- `N5`: `f00079`
- `N7`: `f00079`
- `N9`: `f00079`
- `N11`: `f00040`

`N11` is the reference case.

Earlier same-index `f00080` comparison tables were archived under `/data/Nek5000_data/results/poly_order_compare/archive/same_index_f00080/`. They are diagnostic only; current analysis should use the time-aligned `t19p5` outputs in `/data/Nek5000_data/results/poly_order_compare/tables/`.

## Slice Extraction

Run the slice extractor from the repository root:

```bash
python scripts/02_extract_midspan_slice.py --case N11 --index 40 --overwrite
```

The default slice mode is `nearest_plane`, which selects the rounded `y` plane closest to midspan. For the current data, this selects `y = 0.75`.

The previous finite-thickness slab behavior is still available:

```bash
python scripts/02_extract_midspan_slice.py --case N11 --index 40 --slice-mode slab --overwrite
```

Slice files are written under `/data/Nek5000_data/postproc/poly_order_compare/slices/<case>/`. Duplicate projected `(x,z)` points are averaged before interpolation in the comparison stage.

## Comparison Fields

Comparison CSV files report relative L2, mean absolute error, and Linf error metrics. Relative L2 remains the primary global comparison metric. Mean absolute error is a supplementary global average-difference metric, while Linf / max absolute error is a local maximum-difference metric and can be sensitive to local extrema.

Run concentration comparison:

```bash
python scripts/03_compare_poly_orders.py --comparison-set t19p5 --field concentration --overwrite
```

The concentration comparison interpolates `C`, computes error metrics against `N11`, and writes a front-position table.

Run velocity comparison:

```bash
python scripts/03_compare_poly_orders.py --comparison-set t19p5 --field velocity --overwrite
```

The velocity comparison interpolates `u`, `v`, and `w`, computes speed magnitude, and reports speed errors plus component relative L2 and mean absolute errors for `u`, `v`, and `w`.

Run pressure comparison:

```bash
python scripts/03_compare_poly_orders.py --comparison-set t19p5 --field pressure --overwrite
```

The pressure comparison interpolates pressure, removes each case's spatial mean on the valid common grid, and compares `p_prime = p - mean(p)` against the `N11` pressure fluctuation.

## Plotting

Generate concentration figures:

```bash
python scripts/04_plot_summary.py --comparison-set t19p5 --field concentration
```

Generate velocity figures:

```bash
python scripts/04_plot_summary.py --comparison-set t19p5 --field velocity
```

Generate pressure figures:

```bash
python scripts/04_plot_summary.py --comparison-set t19p5 --field pressure
```

Figures are written under `/data/Nek5000_data/results/poly_order_compare/figures/<field>/t19p5/`.

## Diagnostics

Velocity and pressure plots may show vertical banding. Diagnostics showed that the velocity banding is already visible in the raw midspan slice data and is not primarily caused by slab extraction, duplicate projected `(x,z)` points, or computing speed from interpolated `u/v/w` instead of direct raw speed interpolation.

Run the velocity stripe diagnostic with:

```bash
python scripts/06_diagnose_velocity_stripes.py --case N11 --index 40
```

Diagnostic figures are written under `/data/Nek5000_data/results/poly_order_compare/figures/velocity/t19p5/diagnostics/`.

## Recommended Workflow Commands

Check configured files:

```bash
python scripts/00_check_files.py
```

Probe one Nek5000 file:

```bash
python scripts/01_probe_nek_file.py --case N11 --index 40
```

Run the full `t19p5` workflow:

```bash
python scripts/05_run_t19p5_pipeline.py --fields concentration,velocity,pressure --overwrite
```

Run comparisons individually:

```bash
python scripts/03_compare_poly_orders.py --comparison-set t19p5 --field concentration --overwrite
python scripts/03_compare_poly_orders.py --comparison-set t19p5 --field velocity --overwrite
python scripts/03_compare_poly_orders.py --comparison-set t19p5 --field pressure --overwrite
```

Run plotting individually:

```bash
python scripts/04_plot_summary.py --comparison-set t19p5 --field concentration
python scripts/04_plot_summary.py --comparison-set t19p5 --field velocity
python scripts/04_plot_summary.py --comparison-set t19p5 --field pressure
```
