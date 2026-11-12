# Phase 4: serialization diagnostics for Tecplot and ParaView

`scripts/42_rewrite_vtu_diagnostic.py` rewrites an **existing VTU only**. It does
not read Nek snapshots, differentiate, project, reconstruct, interpolate,
smooth, merge, clip or evaluate eigenvalues. Scripts 40/41, their production
writer, PVD collections, checksums and strict-resume rules are unchanged.
No preferred production encoding has been established. A successful VTK
round-trip is **not** a successful Tecplot isosurface test.

## Observed reference and legacy converter

The input inspected on 2026-11-08 is now located at

```text
/data/Nek5000_data/results/h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.vtu
```

It is 330,551,953 bytes (about 315.24 MiB), with SHA-256
`793bdfc04b22114f7ad5a8f308407bb3941a41049a325d41d80c017591cd94b4`,
matching its production JSON. Its streamed XML headers report:

| Property | Value |
| --- | --- |
| Dataset | UnstructuredGrid, XML version 0.1, one Piece |
| Byte order / block headers | LittleEndian / UInt32 |
| Compressor | vtkZLibDataCompressor |
| DataArray representation | Inline `format="binary"`, base64; no AppendedData |
| Connectivity / offsets | Int64 / Int64 |
| Cells | UInt8 types, reported range 12..12, VTK_HEXAHEDRON |
| Counts | 13,056,000 points; 9,517,824 cells |
| PointData | Float64 scalar omega_x and lambda_ci |
| FieldData | Float64 scalar TIME_VALUE = 19.50240408807 |
| Sampling / bounds in provenance | GLL10 samples of P7; [0,17] x [0,1.5] x [0,1] |
| Full field ranges | omega_x: -26.675716378939278..21.180835970911055; lambda_ci: 0..9.373902076016394 |

The source was subsequently decoded once to validate its volume geometry and
extract a small diagnostic, not to recompute fields. Its checksum was rechecked
and the reference remains untouched. ParaView loading success and the Tecplot
2022 R2 error on enabling Iso-Surfaces were reported by the user; application
behavior was not independently reproduced here. The cause remains unknown.

The available legacy script is `/home/dai_lab/convert_nek_to_vtu.py`. It uses
`visnek`, `pv.Nek5000Reader`, `set_active_time_point`, and
`mesh.save(output_file, binary=True)` for case N7, frames 1..81. It does not
explicitly configure compression, header precision or integer storage, and it
does not compute these two scientific fields. Its output volume was not
available at the configured `/media/.../tecplot_convert` location. Only the
code can be compared: the same save API does not prove identical files,
spatial fidelity, field definitions or Tecplot behavior across VTK versions.
The legacy script was read, not executed or modified.

## Controlled writer configurations

The installed API inspected and exercised is **VTK 9.6.2 / PyVista 0.48.4**.
PyVista's current binary XML writer sets inline Binary mode and ZLib; the
installed VTK default compression level is 5. The new diagnostic uses
`vtkXMLUnstructuredGridWriter` directly and verifies its actual output headers.

| --profile | Mode | Compression | Header | Connectivity / offsets |
| --- | --- | --- | --- | --- |
| reference | Inline base64 Binary | ZLib level 5 | UInt32 | Int64 |
| appended-zlib | Appended, default raw | ZLib level 5 | UInt32 default | Int64 |
| appended-none | Appended, default raw | None | UInt32 default | Int64 |
| appended-int32 | Appended, default raw | None | UInt32 | Int32, after capacity checks |

Appended profiles also accept `--appended-encoding base64`. Only
`appended-zlib` and `appended-none` accept `--header-bits 64`; their default
UInt32 matches the reference and isolates mode/compression differences.
The conservative profile checks maximum point IDs and offsets against signed
Int32, plus every block and a conservative aggregate payload bound against
UInt32. It never converts coordinates or scalar arrays to float32. Large
arrays that cannot meet the limits fail explicitly; use an Int64/UInt64
appended configuration rather than forcing unsafe narrowing.

Raw appended files contain binary bytes after the XML header and are not
generic well-formed XML documents. VTK supports this representation; the
diagnostic inspector stops before raw payloads. Base64 appended output remains
textual XML. Header inspection streams inline payload text without decoding it;
finding the last inline cell-array tags still requires reading through the
file. `--dry-run` makes no output directories and never constructs a VTK grid.

## Whole-cell subsets and exact preservation

The default selects at most 4,096 existing hexahedra (configurable with
`--max-cells 1..1000000`). Larger limits require passing resource preflight.
It locates a cell incident to the maximum lambda_ci
point, then retains a contiguous original cell-ID window around it. An explicit
zero-based `--seed-cell` overrides this. The suggested isovalue is accepted
only if at least one retained Hex has nodal `min < isovalue < max`. The global
midpoint is kept when it passes that check; otherwise the smallest usable
cell midpoint is selected independently of cell ordering. Float64 rounding is
handled explicitly, and unused points never establish a crossing. The JSON
records `isovalue_crossing_cells`. Checks scan at most 4,096 cells at a time.
A selection without any representable strict cell-local crossing is rejected
before staging, including different constant values on separate cells;
increase the limit or choose another seed. The same requirement applies to
`--full-grid`. This is an isosurface diagnostic subset, not a
replacement for the complete positive-x production dataset.

Only complete original cells are selected. Used points remain in their
original relative order, and connectivity is compactly relabeled by ID.
Coincident element-local points are never merged. Coordinates, nodal values,
cell arrays, numeric FieldData, component labels and active attribute roles
are preserved. `selection.npz` records original point and cell IDs; load it
with `allow_pickle=False`. Selecting cells does not change P7 sampling density
within retained cells and creates no new sample locations.

`--full-grid` explicitly preserves **every** original point/cell, including
unused points. It must not be mistaken for a subset operation. No automatic
"all profiles" or frame-series mode exists. Nonnumeric VTK arrays, ghost masks,
non-Hex cells and multi-Piece input are currently rejected rather than dropped
or converted. The original float64 TIME_VALUE is mandatory; time is never
inferred from a filename. Conflicting source-sidecar checksums/times fail.

Each invocation writes one private staged directory, reads its VTU back,
checks exact coordinates, logical connectivity, offsets, cell types, all data
arrays and metadata, validates cell geometry, then publishes without replacing
an existing directory. Only topology integer width may differ in the Int32
candidate. The JSON sidecar records source/output checksums, source provenance,
writer configuration, XML headers, counts, bounds, dtypes, array fingerprints,
file size, exact read-back outcome and the suggested isovalue. Source files
and sidecars must remain unchanged. Failed staging is retained; it is never
automatically recovered or overwritten. Diagnostic files are never inserted
into a production PVD or adopted by production resume.

## Commands for new small diagnostics

These commands generate one bounded subset from the large source. The other
configurations then read only that small VTU. Choose a **new** output root;
existing destinations always fail. All outputs must be outside Git, original
CFD directories and existing scientific frame directories.

```bash
source_vtu=/data/Nek5000_data/results/h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.vtu
diagnostic_root=/tmp/nek_post_vtu_phase4_manual_new

PYENV_VERSION=research312 python -B scripts/42_rewrite_vtu_diagnostic.py "$source_vtu" \
  --output-dir "$diagnostic_root/reference" --profile reference --max-cells 4096 --dry-run
PYENV_VERSION=research312 python -B scripts/42_rewrite_vtu_diagnostic.py "$source_vtu" \
  --output-dir "$diagnostic_root/reference" --profile reference --max-cells 4096

subset_vtu="$diagnostic_root/reference/diagnostic__reference__base64__h32.vtu"
PYENV_VERSION=research312 python -B scripts/42_rewrite_vtu_diagnostic.py "$subset_vtu" \
  --output-dir "$diagnostic_root/appended-zlib" --profile appended-zlib --full-grid
PYENV_VERSION=research312 python -B scripts/42_rewrite_vtu_diagnostic.py "$subset_vtu" \
  --output-dir "$diagnostic_root/appended-none" --profile appended-none --full-grid
PYENV_VERSION=research312 python -B scripts/42_rewrite_vtu_diagnostic.py "$subset_vtu" \
  --output-dir "$diagnostic_root/appended-int32" --profile appended-int32 --full-grid

# Optional independent time-metadata experiment, on the same small mesh:
PYENV_VERSION=research312 python -B scripts/42_rewrite_vtu_diagnostic.py "$subset_vtu" \
  --output-dir "$diagnostic_root/appended-int32-time" --profile appended-int32 --full-grid --add-time-alias
```

The `--full-grid` commands above apply to the **small subset**, not the 330-MB
original. For a raw/base64 comparison, use another new directory and append
`--appended-encoding base64` to an appended-profile command.

## Prepared frame-79 test files

The current implementation wrote five small, standalone files under
`/tmp/nek_post_vtu_phase4_frame79_tests/final/`. All have 5,762 points and
4,096 genuine 3D hexes. Exact cross-variant VTK comparisons preserve coordinates,
cells, both float64 scientific fields and time. The optional alias adds only
FieldData `time`, exactly equal to TIME_VALUE. JSON files sit beside their VTUs.

| Directory | VTU bytes | Filename |
| --- | ---: | --- |
| reference | 161,594 | diagnostic__reference__base64__h32.vtu |
| appended-zlib | 121,569 | diagnostic__appended-zlib__raw__h32.vtu |
| appended-none | 531,193 | diagnostic__appended-none__raw__h32.vtu |
| appended-int32 | 383,737 | diagnostic__appended-int32__raw__h32.vtu |
| appended-int32-time | 383,926 | diagnostic__appended-int32__raw__h32__time.vtu |

The initial extraction and its original-ID `selection.npz` are preserved in
the sibling `reference/` directory above `final/`. The final five files were
rewritten from that exact subset. Copy the full parent tree to retain the
provenance chain; individual VTUs need no external payload files to load.
The source checksum remains the value listed above. No full-grid variant of
the original, Nek calculation or 81-frame processing was run.

Subset bounds are x=9.25..10.125, y=1.375..1.5,
z=0.8048999905586243..0.9132999777793884. Lambda_ci spans
0..9.373902076016394; use **4.686951038008197** as the common isovalue.
Omega_x spans -4.178407026976446..8.225922813639624.

The user reports that all five prepared small VTUs were successfully loaded
and used for isosurface extraction in **Tecplot 360 2022 R2 on Windows 11**.
This is application evidence for those five small files only, not proof of
full-size VTU compatibility or a reason to change the production writer.
Existing diagnostic VTUs and sidecars remain unchanged. New invocations still
set `tecplot_verified=false`: the rewriter cannot certify an application test
for a newly generated artifact.

Subsequent user tests on Windows 11 with 32 GB RAM report successful loading
and isosurface extraction for both 4,096 and 100,000 Hex cells. The complete
frame 79 (9,517,824 Hex cells) initially loads but produces a VTK Data Loader
error when Iso-Surfaces are enabled. Neither memory exhaustion nor an encoding
defect has been established as the cause. Million-cell compatibility remains
untested, and small-file success does not establish full-size compatibility.

## Proposed million-Hex reference diagnostic

The commands below are proposals, not recorded executions. Use a **new**
external directory with sufficient resources; the default reference profile
remains inline/base64, ZLib, UInt32 headers and Int64 topology. Run the dry-run
first, inspect its estimates, then deliberately run the same command without
`--dry-run`. No full-grid rewrite or CFD computation is involved.

```bash
PYENV_VERSION=research312 python -B scripts/42_rewrite_vtu_diagnostic.py \
  /data/Nek5000_data/results/h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.vtu \
  --output-dir /tmp/nek_post_vtu_phase4_frame79_1m_reference \
  --profile reference --max-cells 1000000 --dry-run
```

If that destination already exists, choose another new directory; nothing
is overwritten. A request retains at most the given number of original whole
cells, so a smaller source yields fewer cells without padding or interpolation.

## Manual Tecplot 360 2022 R2 test

Use a fresh frame/session for each candidate so caches and old zone styles do
not confound results. Keep the original file and all test VTUs unchanged.

1. Record the exact Tecplot build/maintenance release, OS and available RAM.
2. Load one small VTU with the VTK Data Loader. Confirm a 3D Cartesian dataset,
   one volume zone, the expected node/element counts and brick/Hex topology.
3. Confirm nodal omega_x and lambda_ci, coordinate bounds and actual time in
   FieldData/auxiliary data. Do not assign time from `f00079`.
4. In Iso-Surface Details, choose lambda_ci as the isosurface variable and
   specify 4.686951038008197 for the prepared files. For new diagnostics, use
   the JSON `suggested_isovalue`, verified against retained cell ranges.
5. Enable Iso-Surfaces. Color the surface by omega_x, using the same color range
   and camera for each file. Do not smooth, merge or resample the input.
6. Record whether loading and isosurface generation each succeed, the exact
   error text, approximate process memory before/after both actions, and time
   assignment. Take a screenshot if useful.
7. Repeat for the other small configurations. Test the `time` alias separately.
8. In ParaView, independently load the same small files, confirm Information
   counts/types/bounds and both PointData arrays, and compare the same contour
   level/coloring. No application import or rendering is claimed by our VTK test.

Suggested worksheet columns: filename, Tecplot build, encoding, compression,
header/index bits, counts, time assignment, isovalue, load result, iso result,
exact error, load RSS, iso RSS, notes. A loader may defer field reads until
Iso-Surfaces is enabled, so initial load success alone is insufficient.

Tecplot's [VTK loader announcement](https://tecplot.com/2017/11/16/vtk-data-loader/)
documents binary and binary-appended support and brick cells. Its
[current loader manual](https://tecplot.azureedge.net/products/360/current/help/users_manual/data-loading.html)
describes using FieldData `time` or `timevalue` for solution time. These are
format/time documentation, **not independent application-test evidence**.
The optional `--add-time-alias` retains TIME_VALUE and creates only an additional
float64 `time`. Successful isosurface extraction does not establish actual
time assignment.

## Resource limits and interpretation

The 330-MB compressed file represents about **1,217,041,160 decoded array
bytes**, before VTK objects, topology copies, decompression and any surface
geometry. A 32-GB workstation is not evidence that all isosurface operations
will fit. The rewriter must currently decode the full source even for a small
subset; it is not a streaming cell extractor. The million-cell cap is a bound,
not permission to bypass preflight. Header-only planning uses Python integer
arithmetic and at most `min(source_points, 8*selected_cells)` selected points.
Int64 connectivity/offset allowances are used even for Int32 profiles.

Let `D` be decoded source-array bytes, `F` source-file bytes, `O` selected
output-array bytes, `C` selected cells, and `P` selected point upper bound.
For subsets, resident ID mappings use `M=8*(P+C)` bytes. Planning adds:

- Extraction: `O + 512*C + 72*C + M`. The `512*C` allows eight int64 sorting/
  compaction workspaces per cell vertex for `np.unique`; `72*C` covers packed
  nine-integer Hex records. This compaction still sorts up to eight million
  references for a million cells; its runtime has not been benchmarked.
- Writing/read-back: `5*O + 72*C + M`, covering selected and read-back arrays,
  writer buffers, comparison temporaries and private writer topology.
- Total initial RAM: `7*D + 2*F + max(extraction, writing) + 1 GiB`. The existing
  source/VTK/decompression allowance is retained rather than traded away for
  the larger extraction. Full-grid mode has no extraction or ID mappings.
- Disk: `2*O + 1 MiB + 4096*array_count` for the VTU, `2*M + 1 MiB` for subset
  NPZ mappings, plus 128 MiB reserve. This allows base64 and compression
  expansion without assuming a favorable compression ratio. Publication
  renames the staged directory and does not duplicate its payload.

For the documented frame-79 header and its two scalar arrays:

| Requested Hex cells | Point upper bound | Output-array upper bytes | Initial available RAM required | Free disk required |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 800,000 | 39,300,008 | 9.75 GiB | 0.214 GiB |
| 1,000,000 | 8,000,000 | 393,000,008 | 11.51 GiB | 0.993 GiB |

The million-cell VTU file allowance is 787,077,264 bytes; its actual compressed
size is unknown. Sharing existing point IDs often lowers the actual point
count, but coincident element-local points are never merged to save memory.
Advanced indexing already makes owned selected arrays; redundant copies have
been removed, and packed/inverse topology temporaries are released before
copying scientific fields. Serialization settings remain unchanged.

Before decoding, storage/header limits, current available RAM (including cgroup
and process address-space limits), free disk and the process file-size limit
are checked. RAM/disk/file-size headroom is checked again before extraction
and before creating output staging. After decoding, the remaining requirement
is `max(extraction, writing) + 1 GiB`; the resident source is already reflected
in available memory. Conservative pre-decode point/block upper bounds can
reject a request that an exact compacted count might fit. Errors explain the
limiting resource; reduce `--max-cells` or use another host/filesystem.
These are conservative formulas, not measured peaks or resource reservations.

All retained samples remain float64. Narrowing integer topology can reduce
file size and memory, but cannot change the physical fields. UInt32 headers
are guarded against data-block and aggregate limits; larger blocks require
an explicit UInt64 appended profile. The process file-size limit is checked;
filesystem-specific maximums, quotas, loader internals and GPU memory remain
unestablished. Files must not be modified concurrently; staging/identity checks
do not freeze an uncooperative writable directory after the last check.

Increase diagnostic sizes deliberately, keeping the original intact; this
change does not automatically generate a million-cell or full-grid file.
If small files work but a full file fails, investigate
memory/size, deferred reads and application-specific behavior; this does not
prove an encoding defect or OOM. A partitioned VTU/PVTU or VTM experiment may
be a later step; none is implemented here. If only certain small configurations
work, that is evidence for isolating serialization differences, not permission
to change the production writer automatically.

Future integration requires a verified Tecplot isosurface result, ParaView
validation, and a deliberate production change covering binary format checks,
array fingerprints, writer configuration provenance, strict-resume signatures,
filenames, atomic staging and PVD behavior. The existing production binary
fingerprint deliberately accepts inline format only, and its FieldData validator
expects only TIME_VALUE. Do not bypass these checks to adopt diagnostics.

## Automated verification

After the million-cell limit extension on 2026-11-08, **166 diagnostic tests
passed in 4.40 s**, including 11 writer
configurations (raw/base64 and UInt32/UInt64 where supported), exact metadata
and component-name preservation, safe Int32 limits without huge allocations,
coincident/discontinuous nodes, time aliases, no-Nek-operator call guards,
resource gates and interrupted/colliding/substituted outputs. Existing crossing
tests cover disjoint scalar bands, all six storage permutations of a connected
three-cell source, independent nonempty VTK contours, unusable selections,
unused points, float64 limits and chunk boundaries. The 52 added cases cover
100,000/100,001/1,000,000 limits with small exact round-trips, large synthetic
header budgets without mesh allocation, invalid limits, early storage/RAM/disk/
file-size failures, repeated headroom checks, process address-space limits,
owned subset arrays and CLI compatibility. The complete suite, including
Phase 1/2/3 regressions, passed **2,582 tests in 110.12 s**, with zero
failures/skips. Compileall, CLI help and tracked/untracked whitespace checks
passed. Production scientific and VTU workflow modules match HEAD.

```bash
PYENV_VERSION=research312 python -m pytest -q tests/test_vtu_compat.py -ra
PYENV_VERSION=research312 python -m pytest -q -ra
PYENV_VERSION=research312 PYTHONPYCACHEPREFIX=/tmp/nek-phase4-pycache \
  python -m compileall -q src scripts tests
PYENV_VERSION=research312 python -B scripts/42_rewrite_vtu_diagnostic.py --help
git diff --check
```
