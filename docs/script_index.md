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
| `scripts/19_plot_leading_edge_evolution.py` | Evaluate the N7 concentration field on a fixed-z uniform periodic x-y grid, extract the rightmost `C=0.1` crossing for every y, and overlay selected leading-edge curves. | Configured N7 `GC0.fNNNNN` snapshots | Leading-edge timeseries and metadata CSVs plus PNG/PDF evolution figures under `/data/Nek5000_data/results/poly_order_compare/leading_edge/N7/` | `python scripts/19_plot_leading_edge_evolution.py --overwrite` |

See [leading_edge_evolution.md](leading_edge_evolution.md) for the physical
definition, spectral-element sampling method, output schemas, and validation
commands.
