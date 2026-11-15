"""Accepted primary destinations. Complete runtime write containment remains gated."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DestinationContract:
    primary: str
    products: str
    companions: str
    unknown: bool


CONTRACTS = {
    'inspect files': DestinationContract(
        'R/inspect/logs',
        'check_files.log; stdout existence report',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'inspect nek-file': DestinationContract(
        'R/inspect/logs',
        'probe_nek_file.log; stdout metadata',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'p-refinement extract-slice': DestinationContract(
        'R/p-refinement/postproc/slices/L',
        'slice_L_fNNNNN.npz; extraction log',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'p-refinement compare-fields': DestinationContract(
        'R/p-refinement',
        'postproc/interpolated/{C,velocity,pressure}; results/tables; comparison log',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'p-refinement plot-summary': DestinationContract(
        'R/p-refinement/results/figures/field/set',
        'field figures; plot_summary.log',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'p-refinement run-t19p5': DestinationContract(
        'R/p-refinement',
        'Union of 02/03/04 destinations; parent execution logs',
        'Union of 02/03/04; postproc/logs/run_t19p5_pipeline.log plus CLI child records',
        False,
    ),
    'diagnostics velocity-stripes': DestinationContract(
        'R/p-refinement/results/figures/velocity/set/diagnostics',
        'raw_speed/raw_u/raw_w scatter, y_distribution, compare_speed_interpolation_methods PNGs',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'p-refinement run-multitime': DestinationContract(
        'R/p-refinement',
        'Union of 02/03/04; parent execution logs',
        'Union of 02/03/04; postproc/logs/run_multitime_pipeline.log plus CLI child records',
        False,
    ),
    'p-refinement collect-errors': DestinationContract(
        'R/p-refinement/results/tables',
        'multitime_error_summary.csv',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'p-refinement plot-errors': DestinationContract(
        'R/p-refinement/results/figures/error_summary',
        'metric_field_vs_time.png',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'p-refinement plot-overlays': DestinationContract(
        'R/p-refinement/results/figures/overlays/set',
        'contour and x/z-profile PNGs',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'p-refinement summarize-errors': DestinationContract(
        'R/p-refinement/results/reports',
        'metric CSVs and teacher_error_summary.md',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'energy closure': DestinationContract(
        'R/energy',
        'timeseries/summary CSVs; optional closure/component figures',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'front history kinematics': DestinationContract(
        'R/front/history/kinematics',
        'timeseries/summary CSVs and optional kinematic figures',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'front history paper-overlay': DestinationContract(
        'R/front/history/paper-overlay',
        'case comparison CSVs, summary, four optional PNGs',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'front history processed-overlay': DestinationContract(
        'R/front/history/processed-overlay',
        'processed summary and three optional PNGs; obsolete-name cleanup',
        'Four OBSOLETE_OUTPUT_NAMES may be unlinked beneath this exact destination with overwrite; no unrelated deletion',
        True,
    ),
    'front concentration detect': DestinationContract(
        'R/front/concentration/L',
        'detected/comparison/summary CSVs, optional plots/frame diagnostics; separate cache',
        'R/cache/front-detection/<native sanitized spec path>; sibling .<cache>.tmp-* and .<cache>.old-<uuid>; optional per-frame diagnostics',
        True,
    ),
    'front concentration paper-overlay': DestinationContract(
        'R/front/concentration/paper-overlay/L',
        'detected_front_overlay_output_paths products',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'front concentration reconstructed-overlay': DestinationContract(
        'R/front/concentration/reconstructed-overlay',
        'reconstructed_xt_output_paths products',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'leading-edge extract': DestinationContract(
        'Without override: R/leading-edge/L or R/h_refinement/leading_edge/selection; with contained `--output-dir D`: exact D, no case/frame append',
        'timeseries, metadata CSV or tagged sampling JSON',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'leading-edge plot': DestinationContract(
        'R/leading-edge/L/figures',
        'leading-edge PNG/PDF',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'leading-edge benchmark': DestinationContract(
        'Parent R; B=R/leading-edge-benchmarks/L; each native run D beneath B; child `--output-root R --output-dir D`',
        'runs/warmups; exact D/L_leading_edge_timeseries.csv and D/L_leading_edge_metadata.csv; stdout/stderr/metrics; root CSVs',
        'runs/workers_N/run_NNN and warmups/workers_N/run_NNN; stdout.log, stderr.log, metrics.txt; two root CSVs; warmups removed after successful warmup phase',
        False,
    ),
    'integrate gll-field': DestinationContract(
        'R/integrate/gll-field',
        'L/field/direction/L_fNNNNN_field_integrate_direction.npz',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'cantero equivalent-height': DestinationContract(
        'R/cantero/equivalent-height',
        'L/L_fNNNNN_cantero_equivalent_height.npz',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'cantero mean-front': DestinationContract(
        'R/cantero/mean-front',
        'native cantero_mean_front_timeseries_path CSVs',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'cantero reconstruct-front': DestinationContract(
        'R/cantero/reconstructed-front/N7',
        'cantero_front_reconstruction_output_paths products',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'cantero overlay-re3450': DestinationContract(
        'R/cantero/overlay-re3450 or R/h_refinement/cantero_re3450',
        'native output path set, optional finest-reference tables',
        'Native products above and process caches under R/.cli; no additional native lock/backup protocol asserted',
        True,
    ),
    'h-refinement inventory': DestinationContract(
        'None: stdout only',
        'in-memory JSON report',
        'None: stdout JSON',
        False,
    ),
    'h-refinement extract-slice': DestinationContract(
        'R/h_refinement/slices',
        'snapshot_NNN.npz and snapshot_NNN.json',
        'snapshot_NNN.npz and .json pairs; no native staging/locking claimed',
        True,
    ),
    'h-refinement compare-fields': DestinationContract(
        'R/h_refinement/field_comparison',
        'inventory/run JSON, snapshot trees, field_errors/selected_times CSVs, plots',
        'inventory.json, run.json; snapshot_NNN/{field_errors.csv,selected_times.csv,comparison_grids.npz,metadata.json,difference PNGs}; aggregate CSVs/error_history.png',
        True,
    ),
    'h-refinement convergence': DestinationContract(
        'R/h_refinement/convergence_analysis',
        'analysis JSON, diagnostic CSV tables and PNGs',
        'analysis.json; directional_mesh_resolution/local_refinement_ratios/pairwise_differences/convergence_classification/morphology_diagnostics/translation_diagnostics/finest_reference_errors CSVs; plots',
        True,
    ),
    'h-refinement leading-edge compare': DestinationContract(
        'R/h_refinement/leading_edge/step6b',
        'raw/L/{frames,metadata,statistics}; comparison CSV/NPZ/JSON/figures',
        'raw/L/frames, metadata.json, frame_statistics.csv; comparison/{tables,aligned_fronts.npz,summary.json,PNG/PDF}',
        True,
    ),
    'h-refinement leading-edge plot-refined': DestinationContract(
        'R/h_refinement/leading_edge/refined_gll_nodes10/figures',
        'case/method PNG/PDF and h_refinement_refined_gll_nodes10_report.json',
        'case/method PNG/PDF and report JSON; optional Matplotlib cache under R/.cli',
        True,
    ),
    'lambda-ci native': DestinationContract(
        'R/lambda-native',
        'C/F native VTR/PVD/JSON',
        'Native frame files under C/F; no staging claimed for this writer',
        True,
    ),
    'lambda-ci tensor-slices': DestinationContract(
        'R/lambda-slices/C/F/plots',
        'PNG/PDF/JSON, optional NPZ; sibling slices, case PVDs/preset',
        'If export: C/F/slices/*.vtr; C/*_series.pvd; persistent .<PVD>.lock and same-directory unique .tmp; optional samples/preset',
        False,
    ),
    'lambda-ci tensor-slice-series': DestinationContract(
        'R/lambda-series/C',
        'frames, series PVDs, logs, .stage5-staging and .stage5.lock',
        'C/.stage5-staging/F.<uuid>; C/.stage5.lock; C/logs/*.jsonl and *_summary.json; per-PVD locks/temporaries and atomic JSON temporary',
        False,
    ),
    'omega-x native': DestinationContract(
        'R/omega-native',
        'C/F artifacts; case staging/backups/lock',
        'C/.omega-x-staging/F.*; C/.omega-x-backups/F.<uuid>; C/.omega_x.lock via native PVD-lock helper',
        False,
    ),
    'omega-x validate-series': DestinationContract(
        'None: stdout report only',
        'validated/invalid report; no application output',
        'None: report/progress only; no application allocation',
        False,
    ),
    'omega-x publish-series-pvd': DestinationContract(
        'R/omega-series-pvd',
        'C_omega_x_series.pvd and same-directory private temporary',
        'R/omega-series-pvd/.<PVD>.*.tmp; explicit final PVD directory',
        False,
    ),
    'omega-x saved-scalar-slice': DestinationContract(
        'R/omega-p7',
        'C/F VTR/validation JSON/PNG/PDF; sibling staging',
        'R/omega-p7/C/.F.staging.* beside final F',
        False,
    ),
    'omega-x saved-scalar-series': DestinationContract(
        'R/omega-p7-series',
        'C frames/logs/controls/staging/locks/full-series PVD',
        'C/.omega-x-p7-staging/F.*; C/logs/control JSON temporaries; persistent series-wide C/.omega_x_p7_slice_series.lock; PVD private same-directory temporary plus hard-link/no-replace publication; no additional per-PVD lock',
        False,
    ),
    'vortex export': DestinationContract(
        'R/vortex',
        '[gllN/]C/F binary VTU+JSON; sibling staging',
        '[gllN/]C/.vortex-3d-fNNNNN-* beside F',
        False,
    ),
    'vortex export-series': DestinationContract(
        'R/vortex-series',
        '[gllN/]C frames, series controls, logs, staging/locks',
        '[gllN/]C/series and logs; sibling private frame staging within C; .vortex-3d-series.lock; series PVD locks/temporaries; control JSON temporaries',
        False,
    ),
    'diagnostics vtu-rewrite': DestinationContract(
        'R/vtu-diagnostic',
        'diagnostic VTU/JSON and source-ID subset arrays; sibling staging',
        'R/.vtu-diagnostic-* beside R/vtu-diagnostic',
        False,
    ),
    'results inventory': DestinationContract(
        'None: stdout report only',
        'in-memory catalog; inventory_only is not scientific validation',
        'None: metadata reads and stdout',
        False,
    ),
    'results catalog build': DestinationContract(
        'R/catalog',
        'catalog.json, datasets.csv, conditional artifacts.csv',
        'Fresh R/catalog directory; conditional artifacts.csv; partial directory can remain',
        False,
    ),
    'results navigation preview': DestinationContract(
        'R/navigation-preview',
        'catalog files plus README.md; candidate V for later publication',
        'Fresh R/navigation-preview with catalog.json, datasets.csv, optional artifacts.csv and README.md',
        False,
    ),
    'results navigation publish': DestinationContract(
        'P (two-file exception); operational records under R',
        'P/README.md and P/results_index.csv only',
        'V is read-only; only P/README.md and P/results_index.csv are external write targets',
        False,
    ),
    'leading-edge plot-refined': DestinationContract(
        'R/leading-edge/refined/figures',
        'two method PNG/PDF pairs and N7_refined_gll_nodes10_evolution_report.json',
        'two method PNG/PDF pairs plus exclusive JSON report; optional Matplotlib cache under R/.cli',
        True,
    ),
}

READ_ONLY = frozenset({"h-refinement inventory", "omega-x validate-series", "results inventory"})
