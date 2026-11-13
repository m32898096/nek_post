# Research results catalog: foundation and navigation publication

## 研究者閱讀入口（Phase 2C）

- [網格細化、omega_x、lambda_ci 與三維渦場](h_refinement.md)
- [多項式階數、能量與前緣分析](poly_order_compare.md)
- [歷史、診斷與不確定分類](historical_results.md)
- [分析與 producer-script 證據](producer_scripts.md)

先以研究問題選擇指南，再用已發布 CSV 的 study、analysis、case_id、variant、notes
查找細節。VERIFIED FROM EXPLICIT METADATA 表示控制檔直接支持；
VERIFIED FROM SOURCE CODE 表示目前實作支持；INFERRED FROM CONVENTIONS 表示名稱或
路徑線索；UNKNOWN 表示證據不足。四者不可互換。
指南位於 Git repository；外部 README/CSV 在 Phase 2C 保持唯讀，並未同步發布新內容。
既有 Traditional Chinese 根目錄模板保留原樣，避免與已發布副本分歧。

Phase 2A provides a read-only discovery API and CLI for
`/data/Nek5000_data/results`. The inspected tree on 2026-11-13 contains 2,556
files and 40,760,621,218 logical bytes across `h_refinement` and
`poly_order_compare`. A catalog groups related outputs so that figures, tables,
case runs and volume series can be found together.

## Building and searching

Run from the repository root. Select a fresh output directory outside the
scientific results tree for every refresh.

```bash
PYENV_VERSION=research312 python -B scripts/43_build_results_catalog.py \
  /data/Nek5000_data/results --output-dir /tmp/results-catalog-review --artifacts
```

The command creates `catalog.json`, `datasets.csv` and optional `artifacts.csv`.
An existing destination causes failure. The JSON is the canonical versioned
representation. CSV files use UTF-8, standard quoting and JSON arrays in list
cells. Empty CSV cells correspond to JSON null. An interrupted publication can
leave a partial output directory for inspection. Retry using a fresh path.

The Python entry points are `build_catalog(root, include_artifacts=True)` and
`write_catalog(catalog, output_dir, results_root=root)` in
[results_catalog.py](../../src/nek_post/results_catalog.py). The API returns
rows in stable path order, without wall-clock generation timestamps.

```python
import json
from pathlib import Path

catalog = json.loads(Path('/tmp/results-catalog-review/catalog.json').read_text())
for dataset in catalog['datasets']:
    if dataset['analysis'] == 'convergence_analysis':
        print(dataset['relative_path'], dataset['entrypoint_file'])
for artifact in catalog['artifacts']:
    if '/f00079/' in artifact['relative_path']:
        print(artifact['relative_path'], artifact['dataset_id'])
```

Resolve every `relative_path`, `entrypoint_file`, `manifest_path` and
`metadata_sources` member against the original results root. Optional artifact
rows join dataset rows through `dataset_id`. They preserve individual file
paths, including duplicate basenames in different directories.

## Schema version 1

The top-level JSON contains `schema_version`, `datasets`, `artifacts`,
`file_count`, `logical_bytes`, and `warnings`. Counts include regular files,
including lock/control files, except the reserved root-level `README.md` and
`results_index.csv` navigation artifacts. These two names are excluded regardless
of their contents; exclusion does not establish ownership or permit replacement.
Same-named files below the root remain inventoried. Symlinks are excluded and reported. Empty roots
produce empty lists and zero totals. Unreadable directories/files fail the
scan when encountered through file inspection. Concurrently changing trees
require a later refresh.

| Dataset column | Meaning and source |
| --- | --- |
| `dataset_id` | `ds_` plus the first 20 SHA-256 hex digits of the bundle's relative directory. Stable across content changes at that directory |
| `study`, `analysis` | First two directory components, retained as navigation labels |
| `case_id` | A case-shaped directory label such as `case_N7_H` or `N7`. Identity remains explicitly unverified |
| `variant` | Remaining bundle directory components, including method/run/case labels verbatim |
| `artifact_class` | Directory-label classification, one of `historical`, `diagnostic`, `smoke_test`, `unclassified` |
| `status` | `unknown`, `reported_completed` from run-level control metadata, or `incomplete_or_ambiguous` for broken PVDs, failed/unsupported output references, parser-depth failures, or non-completed/conflicting run statuses |
| `field_names` | Explicit `output_selection.fields`, `field_names`, or singular `field` metadata |
| `polynomial_degree` | Explicit `reconstruction.source_polynomial_degree` or `polynomial_degree` |
| `sampling_method` | Explicit `reconstruction.interpolation` or `sampling_method` |
| `sampling_nodes` | Explicit `configuration.sampling_nodes` or `sampling_nodes` |
| `projection_method` | Explicit `projection_method` metadata |
| `frame_count` | Count of distinct observed physical timesteps. Multiple slices/parts at one time count once. Source frame-index equivalence remains unverified |
| `actual_time_start`, `actual_time_end` | Minimum/maximum finite PVD timesteps. Without a PVD, bounds cover inspected top-level JSON `time` observations only, with a coverage note |
| `relative_path` | Bundle directory relative to the scanned root |
| `entrypoint_file` | Preferred series-level PVD, then other PVD, then first CSV/figure/array/volume in lexical order, then other regular file |
| `manifest_path` | Unique `manifest.json` in the dataset control scope. Multiple manifests leave this unknown with a diagnostic |
| `producer_script`, `code_revision` | Explicit identically named metadata keys only. Source-file fingerprints are retained in original metadata and do not identify a Git revision |
| `validation_level` | Always `inventory_only`. The catalog checks navigation references and metadata, with scientific numerical validation remaining outside this scan |
| `file_count`, `logical_bytes` | Number and sum of regular-file logical lengths owned by the bundle |
| `metadata_sources` | Paths of successfully inspected JSON/PVD controls |
| `notes` | Missing/conflicting metadata, partial coverage, directory grouping assumptions and reported-completion qualifications |

Optional artifact columns are `dataset_id`, `relative_path`, `file_type`,
`logical_bytes`, `mtime_ns`, `role`, and `related_controls`. Roles are
`collection`, `metadata`, `figure`, `table`, `scientific_artifact`, or `other`.
Roles derive from extensions solely for navigation. Related controls include
local-directory JSON/PVDs plus bundle manifest, processing summary and
series-level PVDs. `mtime_ns` is filesystem modification time, independent of
physical simulation time.

Unknown values remain JSON null, including producer/revision/degree fields
without suitable evidence. Explicit conflicting observations leave the field
unknown and produce a note. Historical paths receive a historical class while
retaining their independent completion status. Smoke and diagnostic labels
also carry no scientific validity judgment.

## Grouping and inspection limits

The recognized series analysis directories are `lambda_ci`, `omega_x`,
`omega_x_highorder_slices`, `vortex_3d_series`, and `spectral_volume`.
An exact `manifest.json` establishes an independent run scope even beneath an
existing ancestor collection. A series-level PVD also anchors its enclosing
directory, with a `series` control subdirectory mapped to its parent. The most
specific anchor owns descendants. A parent PVD may remain an aggregate
collection over independently indexed children, with this relationship recorded
in its notes. File inventory ownership stays exclusive, so the aggregate does
not count child artifacts again.
Without an anchor, frame folders named `f` followed by digits group under their
run parent. Other files group by their immediate parent directory. Archive,
smoke and diagnostic paths retain leaf-directory ownership to avoid mixing
logs and trials into production runs. Every regular file has exactly one owner.

Leaf-directory bundles are provisional scientific associations. Multicase
energy-budget tables share a bundle, while a separate figures directory can
remain a separate bundle. Cross-directory study associations require later
curation. IDs depend on this documented grouping rule and path identity.
The nested-run correction preserves schema version 1 and all existing IDs
for unaffected layouts. Future grouping-rule changes require schema/version review.

JSON/PVD inspection reads at most 1 MiB per control, and at most 64 JSON files
per dataset. Exact manifest/processing-summary names receive priority. Larger
controls, unsupported JSON shapes, conflicting scientific fields and malformed
PVDs generate notes. Parser-depth failures are caught explicitly, reported by
control path and excluded from successfully inspected metadata sources. Other
controls and unrelated datasets continue through inspection. JSON lists used for color maps remain artifact inventory
entries and receive an unsupported-object note. Metadata coverage can therefore
be partial even for a reported-completed run.

PVD references are resolved relative to their collection and checked against
the scanned regular-file inventory. Absolute references, outside-root paths,
symlink aliases, missing targets and nonfinite times are rejected. Relative
`../` references that remain within the root are supported. The scanner skips
symlinks and directory symlinks. Results roots supplied directly as symlinks
are rejected. It reads no VTU/VTR/VTI/NPZ payloads and performs no volume hashes.
Directory-label heuristics support navigation without establishing physical
case identity, Reynolds number, polynomial degree, or solver equivalence.

The Phase 2A output writer protects the scanned root and the actual production
root, including their resolved aliases. It exclusively creates a new directory.
The separate Phase 2B publisher below has a narrow root-only interface for the
two authorized navigation files; the Phase 2A writer's guards remain in force.

## Verified research navigation examples

All following paths are relative to `/data/Nek5000_data/results` and were
verified during Phase 2A.

| Research product | Path |
| --- | --- |
| N7_H frame-79 GLL10 3D vortex output | `h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.vtu` |
| Its per-frame numerical/configuration metadata | `h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.json` |
| 81-frame vortex entrypoint | `h_refinement/vortex_3d_series/gll10/case_N7_H/series/case_N7_H_vortex_3d_series.pvd` |
| Native omega_x volume series | `h_refinement/omega_x/case_N7_H/case_N7_H_omega_x_series.pvd` |
| Reconstructed lambda_ci midplane slices | `h_refinement/lambda_ci/case_N7_H/case_N7_H_lambda_ci_y_midplane_series.pvd` |
| H-refinement convergence table | `h_refinement/convergence_analysis/finest_reference_errors.csv` |
| N5, N7 and N9 energy-budget time series | `poly_order_compare/energy_budget_closure/N5_energy_budget_closure_timeseries.csv`, `N7_energy_budget_closure_timeseries.csv`, `N9_energy_budget_closure_timeseries.csv` in that directory |
| Cantero comparison figures | `poly_order_compare/cantero_re3450_multicase/cantero_re3450_multicase_linear_overlay.png` and `cantero_re3450_multicase_loglog_overlay.png` in that directory |

Frame 79 has metadata time `19.50240408807`. The filename index alone supplies
no physical time evidence. The catalog's series records span 0 to 20 with 81
observed timesteps for the inspected vortex, omega_x, lambda_ci, P7 omega_x
slices and spectral-volume PVDs. Run-level completion metadata is available
for vortex and lambda_ci. Other series retain unknown run completion.

Source conventions were cross-checked in `src/nek_post/paths.py`,
`scripts/24_compute_cantero_mean_front.py`,
`scripts/26_overlay_cantero_re3450_multicase.py`,
`scripts/30_analyze_h_refinement_convergence.py`, and existing series docs.
These conventions support navigation examples. Producer assignments stay
unknown unless the scanned metadata explicitly identifies a producer script.

## Initial real-tree findings and Phase 2B

The initial scan groups the 2,556 files into 94 provisional datasets, including
12 smoke-test bundles, 7 diagnostic bundles and 5 historical bundles. Both
studies contain Cantero products. Other categories include field comparison,
leading-edge methods, front detection/kinematics, convergence, combined x-t
plots, energy budgets, spectral volumes, figure collections and report/table
archives. The nested duplicate `case_N7_H` directory under native omega_x is
preserved and reachable through its existing collection.

Unknowns include most producer revisions, direct completion evidence for older
outputs, and cross-directory figure/table associations. Oversize relocation
metadata, per-run JSON inspection caps and color-map JSON lists are reported
explicitly. No historical artifact is automatically retired.

Phase 2B provides curated research navigation over these provisional groups.
The CSV retains the Phase 2A grouping, IDs and schema; the guide supplies
cross-directory links without relabeling groups as validated scientific runs.
Manifest-managed production directories retain their existing file identities.

## Phase 2B preview and publication

The published artifacts are external files, outside this Git checkout:

- `/data/Nek5000_data/results/README.md`: the Traditional Chinese guide from
  [navigation_zh_tw.md](navigation_zh_tw.md). Its relative links are interpreted
  at the results root, rather than at the template's repository directory.
- `/data/Nek5000_data/results/results_index.csv`: dataset rows using the
  unchanged 25-column schema version 1 above.

The [publication CLI](../../scripts/44_publish_results_catalog.py) reuses
`build_catalog` and `write_catalog` through
[results_catalog_publication.py](../../src/nek_post/results_catalog_publication.py).
These are the initial generation commands used from the repository root:

```bash
PYENV_VERSION=research312 python -B scripts/43_build_results_catalog.py \
  /data/Nek5000_data/results --output-dir /tmp/phase2b-catalog-baseline-20261113 --artifacts
PYENV_VERSION=research312 python -B scripts/44_publish_results_catalog.py \
  /data/Nek5000_data/results --preview-dir /tmp/phase2b-navigation-preview-20261113 \
  --readme-source docs/results_catalog/navigation_zh_tw.md
```

The preview directory must be new and outside both the scanned root and the
production scientific root. It contains the usual `catalog.json`, `datasets.csv`,
`artifacts.csv`, plus `README.md`. The preview CSV is published under the name
`results_index.csv`. Review the complete guide, CSV and JSON diagnostics before
requesting publication. Scientific payloads are never opened by these tools.

The initial preview review found that the energy-budget summary contains only
N7, although three separately named N5/N7/N9 time-series tables exist. The guide
states this limitation and leaves the generic closure figure's provenance
unverified. A fresh final preview incorporates that clarification:

```bash
PYENV_VERSION=research312 python -B scripts/44_publish_results_catalog.py \
  /data/Nek5000_data/results --preview-dir /tmp/phase2b-navigation-preview-20261113-final \
  --readme-source docs/results_catalog/navigation_zh_tw.md
PYENV_VERSION=research312 python -B scripts/44_publish_results_catalog.py \
  /data/Nek5000_data/results --preview-dir /tmp/phase2b-navigation-preview-20261113-final --publish
```

Publication rebuilds the full catalog, including artifact paths, sizes and
nanosecond mtimes, and compares it with the preview JSON. It checks CSV bytes
against the same serializer and validates the guide's simple inline local
Markdown links. Every linked target must exist within the root without symlink
aliases; `results_index.csv` is the sole allowed future target. The template
uses no external links, link titles, query strings, fragments or reference-style
links; the validator supports the inline form used here. It does not infer
scientific meaning from link text. Scientific role and case claims require
manual control/source inspection, as performed for this guide.

Both root targets must be absent, including directories and dangling symlinks.
Each is created with exclusive binary mode, with no overwrite or removal.
Only `README.md` and `results_index.csv` are written. Publication beneath the
actual production root is rejected unless the target is that root itself.
A failure between writes
can leave a partial catalog pair for inspection: this is not an atomic pair
transaction, and a retry refuses any existing target. Do not remove an existing
file simply to bypass collision protection.

### Refresh and verification

For a refresh, choose fresh external preview paths and rerun preview generation.
Compare dataset IDs, paths, statuses, diagnostics and scientific-file inventory
with the previous preview. Validate all root-relative links with
`validate_navigation(text, results_root)` and resolve CSV `relative_path`,
`entrypoint_file`, `manifest_path`, and each `metadata_sources` path at the root.
Scientific sources should be quiescent during review and publication. Inventory
checks can detect observed changes, but do not lock out concurrent external
writers or authenticate historical file contents.

There is currently **no in-place refresh/overwrite mode**. With either published
target present, `--publish` stops without replacing it. Publishing a replacement
requires a separately reviewed and authorized catalog-specific replacement
procedure. Preview generation remains available after publication and excludes
the two reserved root artifacts. It should reproduce the same dataset IDs,
counts, rows and CSV bytes when scientific sources have not changed.

The first publication baseline is 2,556 original files totaling 40,760,621,218
logical bytes. `/tmp/phase2b-original-inventory.json` records every original
relative path, size and mtime for the before/after check. The navigation snapshot
contains 94 provisional datasets, 187 diagnostic messages, 92 unknown statuses
and 2 reported-completed statuses; all 94 have `validation_level=inventory_only`.
The diagnostic count includes informational grouping and timestep notes.
Oversize relocation JSON, color-map arrays and partial JSON coverage remain
visible. Unknown producers, revisions and unsupported historical schemas remain
unknown; publication confers no additional scientific validation.

Adding these root files is outside the inspected workflows' strict frame and
collection inventories. For example,
[vortex_3d_series.py](../../src/nek_post/vortex_3d_series.py) checks exact frame
contents, its `series` control directory and explicit numerical module hashes;
[lambda_ci_series.py](../../src/nek_post/lambda_ci_series.py) protects `f*/**/*`
under its run root;
[omega_x_p7_series.py](../../src/nek_post/omega_x_p7_series.py) discovers native
frame sidecars and checks exact triplets;
[omega_x_series_pvd.py](../../src/nek_post/omega_x_series_pvd.py) inventories
`.vtr`, `.pvd` and `.json` files in its native input tree. The reserved `.md` and
`.csv` root files are outside these scopes. No production code, signature,
manifest, checkpoint or PVD reference is changed. This inspection does not
establish compatibility with arbitrary external tools that inventory all root
files; such tools need their own review.

After publication, compare every original baseline tuple, require exactly the
two new root files, and repeat the read-only scan into a fresh external output
directory. Physical file count becomes 2,558; the scientific inventory remains
2,556 files and the dataset count remains 94. Compare subsequent `datasets.csv`
bytes with the published `results_index.csv`. PVD timestamps and paths remain
as originally stored. This metadata inventory comparison does not replace
scientific payload hashing or executable-to-executable solver validation.

### Initial publication verification: 2026-11-13

Publication succeeded with only the two authorized new root files. All 2,556
original paths, sizes and nanosecond mtimes matched the recorded baseline.
The physical total is 2,558 files. The post-publication CLI scan used:

```bash
PYENV_VERSION=research312 python -B scripts/43_build_results_catalog.py \
  /data/Nek5000_data/results --output-dir /tmp/phase2b-catalog-after-publication-20261113 --artifacts
```

The post-publication catalog equals the complete pre-publication catalog,
including artifact ownership and metadata. Its CSV equals the published CSV
byte for byte, with SHA-256
`ad17998b434a9c553fcde26eeca541b02487ddeb5e86ac0d28161e20476e50a0`.
All 42 guide links and 568 references in 88 PVDs resolved. All 81 vortex manifest
records matched frame metadata times and sidecar hashes, with their declared
output paths present. No large volume payload was hashed for this task.
A second `--publish` invocation refused the existing README without writing.

The 73 catalog tests passed in 0.35 seconds; the full suite passed 2,779 tests
in 102.43 seconds. Syntax/import checks, both CLI help commands, CSV schema,
Markdown links, Git whitespace and new-file whitespace checks passed.
Temporary audit evidence is in `/tmp/phase2b-publication-verification.json`;
temporary previews and inventories are local working evidence, rather than
authenticated historical records or committed external artifacts.

## Phase 2A validation

The catalog suite passed 23 synthetic cases. Together with existing path tests,
the focused run passed 32 cases. The full repository suite passed 2,729 tests
in 105.95 seconds. Syntax/import checks, CLI help, local example paths and
new-file whitespace checks passed. The real-tree scan accounted for every
regular file exactly once. All 2,556 output paths, sizes and nanosecond
modification times matched the pre-Phase-2A snapshot. Large scientific payloads
were excluded from content hashing and parsing.


## Supported declared output references

Inspected JSON controls support the following explicit file-reference forms.
Relative paths resolve against the directory containing that JSON control,
including controls under `series/`. Absolute paths are accepted only inside
the scanned results root with no symlink aliases.

| Field | Supported shape |
| --- | --- |
| `output_paths` | One path string or a flat role-to-path mapping |
| `outputs` | A list of path strings |
| `output_file` | One path string |

Null fields and null mapping/list entries denote absent optional outputs.
All non-null paths must identify regular files in the scan inventory. Missing
and unsafe references produce diagnostics naming the control and field, and
set the dataset status to `incomplete_or_ambiguous`. Unsupported containers,
nested descriptors and non-string entries receive an explicit unsupported
reference-schema diagnostic and the same status. Other fields, including
`source_file`, `output_directory`, checksum maps and revision strings, retain
their original semantics and are excluded from file-output checks.

These checks apply only to the JSON controls within the documented inspection
limits. A control without a supported reference field provides no output-path
verification evidence. `reported_completed` continues to mean a source control
reported completion with no detected navigation-reference failure. Scientific
payload validity and numerical correctness retain `inventory_only` validation.
No output paths or scientific controls are repaired by the catalog.

The correction regressions cover parent aggregate collections with independent
child manifests and frame/run-level collections, differing sampling values,
creation-order permutations, nested native omega_x paths, supported scalar and
structured output references, optional outputs, missing/unsafe/symlink targets,
unsupported historical schemas, ambiguous manifest selection, and deep JSON
with valid neighboring controls. The initial targeted run against the original
implementation failed all 10 selected regression cases. The original source
and failure log were retained under `/tmp` for comparison.


### Correction verification

After the review corrections, the catalog suite passes 45 cases, including
22 additional cases covering the findings and their safety boundaries. The
read-only real-tree comparison retains 94 dataset groups and all 94 dataset
IDs, with the same historical/diagnostic classes, completion states and
scientific metadata. It still accounts for 2,556 files and 40,760,621,218
logical bytes. No declared-output failures were observed among the inspected
real controls. Inspection caps continue to limit that conclusion.

The two nested-run regressions in the original failing selection used both
filesystem creation orders with frame-level child collections. Additional
cases exercise each child's run-level `series.pvd` as well. The scalar and
structured output checks retain explicit control-relative bases and accept
existing in-root absolute paths used by current sidecars. Symlink loops are
reported as unsafe/unresolvable references and do not abort the scan.

The final correction run passed all 2,751 repository tests in 100.98 seconds.
The 45 catalog cases passed in 0.18 seconds. Syntax/import checks, CLI help,
documentation links, Git whitespace checks and explicit whitespace checks for
all untracked Phase 2A files passed. Existing tracked files and the index
remain unchanged, and no scientific result was modified.
