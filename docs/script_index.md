# Script Index

This index lists the repository scripts in numeric order. Run commands from the repository root.

## Setup and inspection

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/00_check_files.py` | Check that configured Nek5000 files exist without reading binary contents. | `config/paths.yaml`, `config/cases.yaml` | `/data/Nek5000_data/postproc/poly_order_compare/logs/check_files.log` | `python scripts/00_check_files.py` |
| `scripts/01_probe_nek_file.py` | Read one Nek5000 file and report basic metadata. | One configured `GC0.fNNNNN` file | `/data/Nek5000_data/postproc/poly_order_compare/logs/probe_nek_file.log` | `python scripts/01_probe_nek_file.py --case N11 --index 40` |

## Slice extraction and field comparison

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/02_extract_midspan_slice.py` | Extract one midspan x-z slice from a Nek5000 field file. | Raw Nek5000 `GC0.fNNNNN` file | `/data/Nek5000_data/postproc/poly_order_compare/slices/<case>/slice_<case>_fNNNNN.npz` | `python scripts/02_extract_midspan_slice.py --case N11 --index 40 --overwrite` |
| `scripts/03_compare_poly_orders.py` | Interpolate slices to a common grid and compare lower orders against `N11`. | Slice `.npz` files | Interpolated `.npz` files and error/front CSV tables | `python scripts/03_compare_poly_orders.py --comparison-set t19p5 --field concentration --overwrite` |
| `scripts/04_plot_summary.py` | Plot field summaries from existing comparison outputs. | Interpolated `.npz` files and error CSV tables | `/data/Nek5000_data/results/poly_order_compare/figures/<field>/<comparison_set>/` | `python scripts/04_plot_summary.py --comparison-set t19p5 --field velocity` |
| `scripts/05_run_t19p5_pipeline.py` | Run extraction, comparison, and plotting for one configured comparison set. | Configured Nek5000 files and comparison-set metadata | t19p5 slice, comparison, table, figure, and log outputs | `python scripts/05_run_t19p5_pipeline.py --fields concentration,velocity,pressure --overwrite` |
| `scripts/06_diagnose_velocity_stripes.py` | Diagnose velocity stripe artifacts from extracted slice data. | Velocity slice `.npz` file | Velocity diagnostic figures | `python scripts/06_diagnose_velocity_stripes.py --case N11 --index 40` |

## Multi-time comparison

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/07_run_multitime_pipeline.py` | Run existing extraction, comparison, and plotting steps across comparison sets. | Configured comparison sets | Per-time slice, comparison, table, figure, and log outputs | `python scripts/07_run_multitime_pipeline.py --comparison-sets t05,t10,t15,t19p5 --fields concentration,velocity,pressure --overwrite` |
| `scripts/08_collect_multitime_error_summary.py` | Collect per-time error CSV files into one summary table. | Per-field error CSV tables | `/data/Nek5000_data/results/poly_order_compare/tables/multitime_error_summary.csv` | `python scripts/08_collect_multitime_error_summary.py --comparison-sets t05,t10,t15,t19p5 --fields concentration,velocity,pressure` |

## Summary and visualization

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/09_plot_multitime_error_summary.py` | Plot error-versus-time figures from the multi-time summary table. | `multitime_error_summary.csv` | `/data/Nek5000_data/results/poly_order_compare/figures/error_summary/` | `python scripts/09_plot_multitime_error_summary.py` |
| `scripts/10_plot_selected_overlays.py` | Plot selected-time qualitative overlays and profiles from existing interpolated outputs. | Interpolated `.npz` files | `/data/Nek5000_data/results/poly_order_compare/figures/overlays/` | `python scripts/10_plot_selected_overlays.py --comparison-sets t05,t10,t15,t19p5 --fields concentration,velocity,pressure` |
| `scripts/11_make_teacher_summary_tables.py` | Create teacher-facing summary CSV tables from the multi-time error summary. | `multitime_error_summary.csv` | `/data/Nek5000_data/results/poly_order_compare/reports/` | `python scripts/11_make_teacher_summary_tables.py` |

## Energy budget diagnostic

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/12_check_energy_budget_closure.py` | Check energy-budget closure from teacher-provided energy-budget files. | `energy_budget.dat` files | `/data/Nek5000_data/results/poly_order_compare/energy_budget_closure/` | `python scripts/12_check_energy_budget_closure.py --cases N5,N7,N9 --target 12 --overwrite` |

## Front-position / Figure 5a diagnostics

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/13_front_kinematics.py` | Analyze front position, front velocity, and reconstructed front trajectory. | `front_simple.dat` files | `/data/Nek5000_data/results/poly_order_compare/front_kinematics/` | `python scripts/13_front_kinematics.py --cases N5,N7,N9 --overwrite` |
| `scripts/14_fig5a_paper_overlay.py` | Compare raw front positions against digitized Cantero Figure 5a paper data. | `front_simple.dat` files and digitized paper CSV | `/data/Nek5000_data/results/poly_order_compare/fig5a_paper_overlay/` | `python scripts/14_fig5a_paper_overlay.py --cases N5,N7,N9 --overwrite` |
| `scripts/15_processed_xt_paper_overlay.py` | Overlay processed reconstructed fronts against digitized Cantero Figure 5a data. | Processed front-kinematics CSV files and digitized paper CSV | `/data/Nek5000_data/results/poly_order_compare/combined_xt_overlay/` | `python scripts/15_processed_xt_paper_overlay.py --cases N5,N7,N9 --overwrite` |
| `scripts/17_fig5a_detected_front_overlay.py` | Compare the GC8950_N7 automatic spectral front with digitized Cantero Figure 5a 3D Re8950 data. | Detected-front timeseries CSV and configured Re8950 paper CSV | `/data/Nek5000_data/results/poly_order_compare/fig5a_paper_overlay/GC8950_N7_Re8950/` | `python scripts/17_fig5a_detected_front_overlay.py --overwrite` |
| `scripts/18_n7_reconstructed_xt_fourway_overlay.py` | Reconstruct Re3450 and Re8950 N7 automatic fronts and overlay both with their matching Cantero Figure 5a datasets. | Two detected-front timeseries CSVs and two configured paper CSVs | `/data/Nek5000_data/results/poly_order_compare/combined_xt_overlay/N7_Re3450_Re8950_reconstructed/` | `python scripts/18_n7_reconstructed_xt_fourway_overlay.py --overwrite` |

## Spanwise leading-edge evolution

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/19_compute_leading_edge_evolution.py` | Evaluate N7 concentration on the fixed-z periodic target grid and extract selected rightmost `C=0.1` leading-edge curves; the configured compute default uses two workers for later frames. | Configured N7 `GC0.fNNNNN` snapshots | Leading-edge timeseries and metadata CSVs under `/data/Nek5000_data/results/poly_order_compare/leading_edge/N7/` | `python scripts/19_compute_leading_edge_evolution.py --case N7 --workers 2 --overwrite` |
| `scripts/20_plot_leading_edge_evolution.py` | Validate existing leading-edge CSV artifacts and render the evolution without rereading Nek snapshots. | Leading-edge timeseries and metadata CSVs | PNG/PDF evolution figures under `/data/Nek5000_data/results/poly_order_compare/leading_edge/N7/` | `python scripts/20_plot_leading_edge_evolution.py --case N7 --overwrite` |
| `scripts/21_benchmark_leading_edge_workers.py` | Measure isolated script-19 runs for workers 1, 2, and 4 with GNU time and require exact CSV artifact equivalence. | Configured N7 snapshots and script-19 compute workflow | Per-run logs/artifacts plus benchmark and summary CSVs under `/data/Nek5000_data/results/poly_order_compare/leading_edge_benchmarks/N7/` | `python scripts/21_benchmark_leading_edge_workers.py --case N7 --start-index 37 --end-index 52 --nx 500 --worker-counts 1,2,4 --repeats 2 --all-frames --overwrite` |

## Directional GLL integration

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/22_integrate_gll_field.py` | Integrate one configured Nek5000 scalar snapshot along physical `x`, `y`, or `z` with the validated GLL directional-integration core. | One configured `GC0.fNNNNN` field file | `/data/Nek5000_data/results/poly_order_compare/gll_directional_integrals/<case>/<field>/<direction>/<case>_fNNNNN_<field>_integrate_<direction>.npz` | `PYENV_VERSION=research312 python scripts/22_integrate_gll_field.py --case N7 --index 79 --field concentration --direction y` |

## Cantero-equivalent-height preprocessing

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/23_compute_cantero_equivalent_height.py` | Compute Cantero Eq. (4.1) local equivalent height by z-direction composite GLL quadrature, then Eq. (4.2) spanwise average by composite physical-y GLL quadrature divided by `Ly`. Neither integral uses uniform interpolation. | One configured `GC0.fNNNNN` field file | `/data/Nek5000_data/results/poly_order_compare/cantero_equivalent_height/<case>/<case>_fNNNNN_cantero_equivalent_height.npz` | `PYENV_VERSION=research312 python scripts/23_compute_cantero_equivalent_height.py --case N7 --index 79` |

## Cantero mean-front definition

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/24_compute_cantero_mean_front.py` | Reuse Phase 1 Cantero Eq. (4.1)–(4.2) composite-GLL equivalent-height preprocessing for each configured snapshot, then locate the first positive-x physical-GLL threshold crossing where `h_bar < delta`; default `delta=0.01`. No uniform-grid interpolation, paper comparison, or reconstruction is performed. | Configured `GC0.fNNNNN` field snapshots | `/data/Nek5000_data/results/poly_order_compare/cantero_mean_front/<case>/<case>_cantero_mean_front_timeseries.csv` | `PYENV_VERSION=research312 python scripts/24_compute_cantero_mean_front.py --case N7 --threshold 0.01 --reference-x 0` |

## Cantero reconstructed x-t comparison

| Script | Purpose | Main input | Main output | Example command |
| --- | --- | --- | --- | --- |
| `scripts/25_reconstruct_cantero_front.py` | Formal N7/Re3450 Phase-3 workflow: composite-GLL spatial z/y integration produces equivalent height, `delta=0.01` produces the mean front, and the established temporal gradient/smoothing/trapezoidal reconstruction produces reconstructed `x-t` before the only paper comparison. The raw Phase-2 `x_F` is never overlaid with paper data. `--case` is intentionally restricted to `N7`; expert input-path overrides remain available. | Phase-2 N7 Cantero mean-front CSV and configured Re3450 Figure-5a CSV | `/data/Nek5000_data/results/poly_order_compare/cantero_front_reconstruction/N7/` | `PYENV_VERSION=research312 python scripts/25_reconstruct_cantero_front.py --case N7` |

See [leading_edge_evolution.md](leading_edge_evolution.md) for the physical
definition, spectral-element sampling method, output schemas, and validation
commands.
