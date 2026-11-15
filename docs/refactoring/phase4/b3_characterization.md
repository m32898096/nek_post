# Phase 4B-3 compatibility characterization

Source baseline `588a7dc39c1adedcd0d4304158a3f9cdf7d02959`.

This additive report accompanies the B3 internal APIs and synthetic tests. The
roadmap requires characterization tests for scripts 03/07 and specifies no
report filename. This report uses the existing Phase 4 documentation directory.
Scientific execution, complete write inventories and independent acceptance
remain deferred.

## Prerequisites and environment

`cli.capabilities` copies the 49 architecture profile associations. Profiles are
conservative discovery inventories, with starred plotting/VTK qualifications
retained. Metadata-only routes can require a smaller set in future adapters after
source/branch characterization. Core covers NumPy, SciPy, PyYAML and pymech. Plot
adds Matplotlib/Pillow, VTK and volume add VTK/PyVista, and selected profiles
require pdftoppm or the native /usr/bin/time path. These scientific dependencies
are optional for base CLI startup and required within their selected profile.
Only PyYAML is currently declared as a package runtime dependency.

The internal checker distinguishes available presence, missing required/optional,
unsupported platform/version, indeterminate discovery/tool behavior, and deferred
native-runtime/G-WRITE acceptance. It imports no scientific module and executes
no tool. Available module specifications establish discovery only. Scientific
version ranges are deferred by ADR-0005. GNU time identity and formatting,
Poppler behavior, Linux renameat2, POSIX fcntl/resource, locking, hard links and
fsync on the actual destination filesystem still require native runtime checks.
No result grants execution authorization. Unknown commands raise KeyError.

`cli.environment.plan_environment` requires the B2 SafetyContext and validates R
and each modeled runtime directory through its existing root/destination API.
It copies inherited values and explicitly records overrides, requested removals,
protected input paths, runtime-only writable paths and cwd. Paths describe the
B2 protected inventory, which may include absent future inputs. This API does
not certify input existence or a complete transitive write set.

The required overrides are TMPDIR/TMP/TEMP under R/.cli/tmp, MPLCONFIGDIR under
R/.cli/matplotlib, MPLBACKEND=Agg for future headless standalone execution, and
PYTHONDONTWRITEBYTECODE=1. Benchmark selection additionally specifies LC_ALL=C.
Caller-selected removals are limited to PYTHONPATH, PYTHONHOME and
PYTHONPYCACHEPREFIX. Other inherited values are preserved. These mappings may
contain secrets and must never be recorded wholesale as invocation provenance.
No directories are allocated and os.environ is untouched. Other library caches
and native same-directory staging require route-specific G-WRITE observation.

No public prerequisite command or flag is specified by B3. Both APIs are internal
and the B1/B2 CLI remains unchanged. Future invocation ordering must be B2 source
resolution, complete inventory, destination derivation and authorization, then
prerequisite checks, allocation, environment establishment, imports and execution.
Incomplete/Unknown plans remain unavailable before allocation.

## Script 03

The designated file is `scripts/03_compare_poly_orders.py`, identified by C4 in
implementation_roadmap.md and the 03 migration-matrix row. `_parse_args` occupies
lines 48–72 and `main` lines 75–277. Its main guard is at the end of the file.
Import loads comparison, IO, selection and interpolation globals, including
NumPy, SciPy griddata and PyYAML through config/paths. Parsing --help occurs after
these imports. Import does not invoke main. Public imported globals remain
monkeypatch targets in the original module namespace.

Arguments are comparison-set, index, case-indices, field (concentration default,
velocity or pressure), method (linear), overwrite (false), time-tolerance (0.05),
strict-time (false), and front-threshold-ratio (0.01). Invocation is the original
Python script entrypoint. It uses REPO_ROOT from __file__ to load config/paths.yaml
and config/cases.yaml. It has no B2 environment configuration selection. Legacy
ProjectPaths converts YAML entries directly to Path, so relative values retain
process-cwd interpretation. Configuration and ProjectPaths construction precede
the main try block. Invalid configuration can propagate before comparison-error
logging. No new script environment variables or subprocesses are introduced.

Inputs are saved postproc/slices/L/slice_L_fNNNNN.npz files for configured ordered
cases. Selection uses comparison_selection.py, preserving named-set reference,
case-specific indices and override precedence. Missing inputs give regeneration
instructions. IO loads NumPy arrays from NPZ. Time alignment uses actual slice
times with reference-relative differences. Strict mismatch logs and exits 1
before grid construction. Input snapshots are read by upstream script 02, whose
Nek reader lazily requires pymech. Script 03 consumes the extracted NPZ files.

interpolation.py constructs the common physical x-z overlap grid, averages
rounded duplicate projected points and applies scipy.interpolate.griddata.
comparison.py retains separate concentration, velocity/speed and pressure-prime
calculations and common finite masks. Sampled metrics retain their existing
meaning. N11 is a configured comparison reference. Existing comparison,
comparison_selection, comparison_io and reporting tests provide synthetic
numerical/schema references.

All branches compute their comparison before checking each existing NPZ. Without
overwrite an existing NPZ is skipped, while CSV error tables and appended
postproc/logs/compare_poly_orders.log still occur. Concentration additionally
writes front-position CSV and named-set metadata text. Velocity writes u/v/w/speed
arrays, pressure writes p/p_prime. Paths are defined by comparison_io.py. Parent
directories are created by writers. CSVs and metadata use replacement writes,
NPZ uses savez_compressed, and logging appends timestamps. Errors inside the try
attempt logging and exit 1. Partial outputs can remain. There is no new locking,
transaction, resume or cleanup protocol. Complete process/cache/failure write
closure remains Unknown.

B3 tests execute all three unchanged main branches with recording fakes, both
existing-NPZ skip and overwrite selection, case-specific indices and config
paths. Kernels and writers are replaced in that module for orchestration tests.
Existing kernel/IO tests remain separate numerical evidence. Migration requires
an explicit-context adapter preserving these globals, ordering and schemas after
B2 authorization and dependency checks. Calling checkout-based main directly
would retain checkout configuration assumptions.

## Script 07

The designated file is `scripts/07_run_multitime_pipeline.py`, identified by C7
and the 07 matrix row. `_parse_args` is at lines 19–39, command helpers at
90–150, and `main` at 153–228. Import loads stdlib, config/paths and PyYAML and
defines functions. Its main guard prevents work on import.

Arguments are comparison-sets, fields (concentration,velocity,pressure), overwrite,
dry-run, skip-slices, skip-plots and strict. Flags default false. Failure always
stops at the first child failure, including without --strict. Field/set lists
preserve caller ordering and duplicates. Default sets use the configured
multitime_comparison_sets with t19p5 fallback. Unknown fields/sets fail.

Configuration is the same checkout-fixed paths.yaml/cases.yaml pair as script
03, loaded before the try block. Child argv uses sys.executable and relative
scripts/02_extract_midspan_slice.py, 03_compare_poly_orders.py and
04_plot_summary.py. subprocess.run receives cwd=REPO_ROOT and check=True. It
inherits the environment through subprocess defaults and uses argument arrays.
No shell is requested. REPO_ROOT determines child cwd independently of the
parent invocation cwd. Original relative YAML entries still follow each process
cwd. These defaults remain unchanged.

Extraction walks selected sets in order, then their case_indices insertion
order, deduplicating first occurrences of (str(case), int(index)). After every
extraction, it walks ordered sets then ordered fields and immediately compares
then plots each pair. Skip flags omit the applicable stages. Overwrite reaches
02/03, while the 04 builder has no overwrite flag. Actual child field parsing,
NPZ skip and plotting behavior remains the child's responsibility.

The parent prints and appends timestamped postproc/logs/run_multitime_pipeline.log
for startup, each command and completion/failure. Legacy dry-run still creates
and appends this log, but skips subprocess execution. Successful child returns
append Success. CalledProcessError exits with the child's code after error
logging. Other exceptions attempt error logging and exit 1. Logs and previous
child outputs can remain after failure. There is no pipeline restart manifest,
rollback or new lock protocol. Complete child/cache write closure remains
Unknown. Upstream raw Nek inputs and downstream figures/CSV/NPZ/logs follow
02/03/04 and their actual selected dependencies, including plotting eager imports.

B3 tests exercise actual builders, first-occurrence dedup, full extraction then
set/field compare/plot order, default-set fallback, legacy dry-run logging,
subprocess cwd/check behavior and first-failure exit propagation. Subprocesses
and log writes are recorded by fakes. Production data and child programs are
never executed by these characterization tests.

## Benchmark compatibility and limits

No benchmark orchestration or hooks are added in B3. The original module retains
its builder, cwd and inherited environment behavior, LC_ALL=C, GNU-time wrapping,
warmup cleanup and measured CSV/equality definitions. Future installed execution
retains parent R, B=R/leading-edge-benchmarks/L and exact child D under B for
warmups and measured runs. Both CSV artifacts belong directly under D. Effective
config files remain immutable shared-R inputs under the B2 explicit internal
run-owned role. C11/C12 must establish actual argv, placement/readback, native
collision and environment behavior before activation.

Import audit hooks observe Python events and explicit imports. They cannot prove
arbitrary C-library or external-tool filesystem behavior. Bubblewrap bounds test
writes to disposable evidence directories. All 31 Unknown sets and route-specific
G-WRITE, G-EXEC, G-LIFECYCLE and scientific acceptance gates remain pending.
