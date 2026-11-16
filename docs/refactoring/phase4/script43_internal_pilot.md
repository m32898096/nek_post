# Script 43 internal implementation pilot

This is an internal metadata-inventory pilot, authorized for supervised finite
synthetic fixtures only. Public `results inventory` and `results catalog build`
remain unavailable. This note does not activate a public route, clear G-WRITE,
authorize production use, or constitute independent implementation acceptance.

## Internal interfaces

All entry points are in `nek_post.cli.script43_adapter`. Keyword-only wrappers
return integer 0 on success and 1 on handled failure. Unknown keywords raise
Python argument errors; no public parser or registry entry was added.

| Interface | Responsibility |
| --- | --- |
| `prepare_inventory(input_root=..., cwd=..., include_artifacts=False)` | Validate explicit input and existing B3 requirements; return frozen `InventoryPlan`, with no YAML or output allocation. |
| `collect_catalog(plan)` | Revalidate supplied/canonical input and prerequisites, then call unchanged `build_catalog` exactly once. Return its plain dictionary without calling the writer. |
| `prepare_catalog_build(input_root=..., output_root=..., cwd=..., paths_config=None, environ=None, include_artifacts=False)` | Resolve original machine paths, build full B2 protections, validate existing R and fresh R/catalog plus possible products, then check B3. Return frozen `CatalogBuildPlan`; do not import the catalog handler or allocate. |
| `publish_catalog(plan, catalog)` | Revalidate configuration identity, input, full safety context and exact destinations; retain the native scientific-root guard; call unchanged `write_catalog`. Return `PublicationResult` or raise chained `PublicationFailure`. |
| `inventory(input_root=..., cwd=..., include_artifacts=False, stdout=None, stderr=None)` | Construct and print the full catalog as JSON; no filesystem publication. |
| `catalog_build(..., dry_run=False, stdout=None, stderr=None)` | Prepare, construct, explicitly publish, then print native summary JSON and `Datasets: N`. Dry-run validates and constructs, prints `NOT PUBLISHED`, and never calls the writer. |

Plans are data for trusted internal callers, not unforgeable permissions. A
caller must pass the unchanged catalog collected for the writing plan. This is
not an API for authenticating arbitrary dictionaries or historical JSON files.
No function changes process cwd, environment, bytecode settings or logging.

## Configuration and capabilities

Writing uses unchanged B2 `resolve_configuration('paths', ...)`: explicit file
over `NEK_POST_PATHS_CONFIG`, otherwise refusal. There is no machine default.
Configuration filenames and explicit roots use captured cwd; YAML path values
use the canonical supplying file's parent. Cases/study YAML is unused.
Whole-file digest verification and rebuilding/comparing the original normalized
inventory occur before publication. All configured cases, historical P/H roots,
references, configuration, input and package/native ancestors stay protected.
There is no selected-case narrowing or run-owned exemption.

Both routes retain B3 core: Python >=3.12, numpy, scipy, yaml, pymech and the
deferred native-runtime-and-G-WRITE result. Required nondeferred failures refuse
execution before allocation. Deferred results keep `execution_authorized=False`.
Package presence is not importability, scientific validity or execution authority.

## Exact publication boundary

R is explicit, existing and real, with stable cooperative ancestors. The only
application products are fresh `R/catalog`, `catalog.json`, `datasets.csv`, and
`artifacts.csv` when the actual artifact list is nonempty. R and its ancestors
are never created. B2 authorizes root and individual destinations separately,
including revalidation immediately before the native writer. Input/R overlap
in either direction, lexical output `..`, existing output symlink components,
and existing D/products are refused. The supplied final input symlink is checked
before normalization; input parent aliases retain native behavior.

The native writer serializes strings, exclusively creates D, and exclusively
opens final files in sequence. Native UTF-8 JSON, insertion order, LF suffix,
fixed CSV columns, CRLF rows, cell quoting, nulls and conditional artifacts are
preserved. No replacement, resume, cleanup, suffix retry, temp, log, cache,
lock, backup or `.cli` allocation is intended. Native modes follow process umask;
Script 00's descriptor allocator and explicit modes are not inherited.

## Failure semantics and compatibility

Partial directories and files remain after failure and block retry. An exception
records phase/code/cause, writer invocation, allocation uncertainty, expected
products, bounded lstat observations and inspection errors. A symlinked D is not
followed to inspect children. Failed inspection is Unknown, not absence. Regular
observed files on failure are unattributed and not certified complete. Successful
publication records expected/published files; a later stream failure returns 1
while reporting `publication_complete=True`. Failures of both streams, signals,
memory exhaustion and cancellation require external supervision; they are not
swallowed as successful execution.

Approved adapter differences include mandatory B2 configuration/normalization,
existing R and fixed D, stricter output aliases/traversal, internal inventory
JSON, dry-run, concise route/phase diagnostics, nonzero failure returns and
explicit partial-state reporting. Native builder, writer and legacy script are
unchanged. Native warnings remain compatible with successful status. Legacy main
return behavior, historical harness process status and adapter return behavior
are separate. Standalone legacy failing-process parity remains unmeasured;
E-EXIT requires separate authorization.

## Required execution controls and remaining limits

Before interpreter imports, the trusted supervisor must suppress bytecode,
protect source/input surfaces, mask production, isolate temporary/cache paths,
deny network/children, limit resources, and observe writes. Fixtures are finite
ordinary files/links: <=100 entries and files, depth <=8, with only the accepted
1,048,577-byte/65-JSON/deep-JSON stress shapes. Immutable manifests and mode,
size, mtime, content/link checks accompany each run. Unreadable contents are
Unknown; atime is excluded. Synthetic test setup/observer writes have separate
scratch authority. The adapter itself supplies neither a sandbox nor a global
traversal limiter.

The verified supervisor uses read-only Bubblewrap mounts, seccomp, nonroot DAC,
`-B -I`, isolated scratch and strace. Limits are CPU 5 seconds, address space
512 MiB, file size 16 MiB, 128 descriptors and 25 seconds wall time. These are
pilot controls, not new B3 definitions or adapter defaults.

Global traversal bounds, hostile filesystem race protection, complete-file
atomicity, multi-file transactions, durability and universal write-set closure
remain unestablished. Ordinary Python TextIO semantics are assumed: an artificial
short-count-without-error writer can still return success with truncated JSON.
Existence/type checks do not authenticate complete bytes. No concurrent or
unattended consumer is allowed; accept output only after successful status and
external expected-file/schema checks. Stronger publication, arbitrary trees,
installed portability, public activation and G-WRITE clearance require separate
work and authorization. All 31 Unknown write-set flags remain unchanged.

## Validation evidence

Implementation reports, commands, source hashes, test records, traces, original
evidence preservation checks and the 164-execution/82-pair regression crosswalk
are retained under `/tmp/nek-post-phase4c-c-script43-implementation/`.
`test_results.md` is authoritative for executed results and exclusions;
`final_integrity.json` identifies the exact local uncommitted source. Independent
implementation review remains a separate next stage. No commit, merge, push,
fetch, pull, protected-ref mutation or scientific workflow is authorized here.
