# Nek5000 source and production provenance

Observed on **2026-11-12** for Phase 1A, from repository baseline
`c63bbd8b7b4e1c823ab7336ff07a7c153676d30d` on
`feature/nek5000-numerical-validation`. See the [methods index](README.md),
[numerical correspondence](vortex_numerical_method.md), and
[Phase 1B specification](phase1b_validation.md).

## Production-linkage status

**UNVERIFIED:** the exact source revision and executable that generated the
configured production simulations, especially `case_N7_H`, remain unidentified.
There is useful historical build and mesh evidence, but no complete chain from
the production snapshots to a source revision, build invocation, and binary.
No commit or build identity is inferred from a directory name, timestamp,
`20.0-dev` banner, matching element count, or the GC_small practice case.

The [evidence inventory](nek5000_evidence.json) pins 40 inspected files by
SHA-256, six small archive members, raw-header observations, and source-line
citations. It is documentation evidence, not a production checkpoint or a
solver-generated reference field. Existing scientific metadata was read without
editing it; copied metadata expressions in the inventory are verbatim evidence.

| Question | Established evidence | Remaining gap |
| --- | --- | --- |
| Available local source | `/home/dai_lab/Nek5000/core/HEADER:12` says `20.0-dev`; seven inspected core files match the second local tree byte-for-byte | No `.git` metadata in either source tree; exact revision unknown |
| Historical build source path | `/data/3D_chnl_pbc/makefile:5` records `/home/ylhuang/Nek5000` | That source tree is unavailable at the recorded path; correspondence to current local core unverified |
| Historical compiler configuration | `makefile:10` records `mpif77` for compiler/linker, `mpicc` for C; flags at line 13 | Exact compiler/MPI versions, resolved optimization settings, and binary-to-makefile relationship unverified |
| Numerical precision | Historical makefile requests eight-byte default real; production headers use four-byte field storage | Internal precision/build flags for each production executable unknown |
| Historical mesh and BCs | Binary mesh plus box/log records identify x/y periodic pairing and z boundaries for the candidate case | Candidate-to-production mesh and face-numbering linkage incomplete; N7_H differs in sampled z geometry |
| Historical binary identity | A candidate executable has SHA-256 and an ELF build ID | No evidence that it generated N7_H, VH, VVH, or the other configured cases |
| Python provenance | Baseline source fingerprints and existing frame-79 postprocessing metadata | Postprocessing signatures do not identify the original simulation solver |

## Search scope and non-execution policy

Targeted discovery inspected source/build/configuration names under the eight
configured raw case directories, `/data/Nek5000_data/postproc`,
`/data/3D_chnl_pbc`, `/home/dai_lab/Nek5000/run/LX_practice`, and both available
local core trees. Targeted filename searches also covered
`/home/dai_lab/Nek5000_post`, `/home/dai_lab/Documents`,
`/home/dai_lab/Downloads`, and `/data/code`. This is not an exhaustive search of
every mounted filesystem or the original remote machine.

The configured raw directories contain snapshots and some descriptors,
diagnostic data, or visualization states. No `SIZE`, `.usr`, `.par`, `.re2`,
`.ma2`, solver executable, or build record was found inside those eight case
directories. Separate candidate files were found under `postproc` and
`3D_chnl_pbc` and are assessed below rather than attributed to production.

Collection read text, 132-byte snapshot headers, the coordinate block of the
first file record in three snapshots, a small binary mesh, small-file hashes,
ELF metadata, and six archive members in memory. It did not execute a solver,
invoke make, extract an archive to disk, build a reference executable, recompute
fields, decode a VTU, run resume, or alter scientific files. Even make dry-run
was avoided: the historical makefile contains parse-time `$(shell ...)`
operations at lines 68 onward.

## Available local source identity

The comparison source root is `/home/dai_lab/Nek5000/core`.
`/home/dai_lab/Nek5000/Nek5000-master/core` is a second available tree. Neither
has `.git` metadata at its root. The following files have identical bytes in
both trees, with full SHA-256 values recorded in the inventory:

| Local file | Main evidence |
| --- | --- |
| `HEADER` | Version banner at line 12 |
| `speclib.f` | `ZWGLL:107`, `ZWGLJD:238`, `DGLL:800` |
| `navier5.f` | `local_grad3:428`, `gradm1:462`, `comp_vort3:512` |
| `postpro.f` | `lambda2:19`, `comp_gije:252`, `filter_s0:343` |
| `coef.f` | GLL tensor weights:260, metrics:488, `bm1`:735, `setinvm`:1500 |
| `dssum.f` | `setupds:1`, `dssum:33`, gather-scatter addition:79 |
| `navier8.f` | `set_vert:3`, periodic numbering and user hook |

These fingerprints identify the source bytes reviewed in Phase 1A. They do not
identify a Git commit or prove these bytes were compiled into production.
No upstream source was substituted for the local implementation.

The earlier [vorticity page](../nek5000_vorticity_algorithm.md) records a dated
upstream-master inspection. Retain that as historical evidence; use this pinned
local-source record for the current comparison.

## Historical channel-case evidence

The stronger non-practice lead is `/data/3D_chnl_pbc/`. It contains case files,
logs, `GC.re2`, `GC.ma2`, an executable, snapshots, and a tar archive.
**CONFIRMED artifact observations do not establish a production assignment.**

### Build and executable

`/data/3D_chnl_pbc/makefile:1` records generation by makenek on
`09/22/2025 09:44:05`. Lines 4 and 5 identify case directory
`/home/ylhuang/chnl_pbc` and source root `/home/ylhuang/Nek5000`. The latter path
is not available on this machine.

The makefile records `FC=LD=mpif77`, `CC=mpicc`, and

```text
-cpp -fdefault-real-8 -fdefault-double-8 -w -std=legacy -mcmodel=medium -fPIC
```

It also names MPI/global-integer preprocessing settings and BLAS/LAPACK/gslib
link libraries. Its `FL2`/`FL3` expressions reference `L2`/`L3`; no resolved
optimization invocation or compiler-version build transcript was recovered for
this candidate. These are recorded makefile settings, not verified N7_H flags.

The candidate `/data/3D_chnl_pbc/nek5000` is an unstripped ELF64 x86-64 executable,
1,950,408 bytes, inspected without execution:

```text
SHA-256: d6b8d7188ba38dd740ce5b48eb7160ddb51d4ff8ae3c25e9ab3627a709de5721
ELF build ID: 1d4434b82b82709aaebb653593a5f9955cf60357
```

Its `.comment` section contains GCC 8.5.0 Red Hat `-3` and `-4` strings. Those
strings can come from linked components and do not recover the full Fortran/MPI
toolchain. An ELF build ID is not a source Git commit.

`GC.log1.80:11` reports `20.0-dev`, line 16 reports 80 MPI ranks, and lines 65/66
report 26,112 elements and `lx1/lx2/lx3/lxd = 6/4/6/9`. `SIZE:13` also sets
`lx1=6`, corresponding to degree five, not N7_H's eight-node P7 representation.

### Mesh and boundary records

`GC.box:17` specifies `272 x 12 x 8` elements. Lines 18/19 give x/y bounds
`[-17,17]` and `[0,1.5]`; lines 21/22 list graded z boundaries. Lines 23/24 give
velocity `P,P,P,P,W,SYM` and temperature `P,P,P,P,I,SYM` boundary codes.

Direct read-only decoding of `GC.re2` confirms:

- Format `#v002`, dimension 3, 26,112 elements, and zero curved-edge records.
- Corner bounds `[-17,17] x [0,1.5] x [0,1]` and 273/13/9 distinct corner
  coordinates along x/y/z.
- Velocity records: 4,544 `P`, 3,264 `W`, 3,264 `SYM`.
- Temperature records: 4,544 `P`, 3,264 `I`, 3,264 `SYM`.
- Every periodic record has a reciprocal partner in each field. Pair-centroid
  displacement identifies 192 directed x records and 4,352 directed y records.

This establishes candidate-mesh boundary evidence. The record layout was checked
against installed Pymech 2.0.1 `neksuite/mesh.py:601` (`readre2`). Reciprocal
records and centroid directions are not a full face-permutation/orientation
certificate. They also do not transfer boundary conditions to another mesh.

The mesh SHA-256 is
`516143a42541a1c5d815fc293f8540a4a244831f6c23eb88059817c679085dee`.
`/data/Nek5000_data/postproc/GC.re2` has identical bytes.

### Configuration conflicts and sampled geometry

The available artifacts are not one immutable run bundle:

- `GC.log1.80:21` records end time 50; loose `GC.par:4` has `2*5e-2`, also
  recorded in the separate `logfile:21`.
- The loose parameter file uses viscosity/conductivity `2.8986e-4`. The tar
  member uses `1.538e-4` and end time 50.
- The archived box has a different signed z-count and boundary-code text from
  the loose box and inspected binary mesh. Do not combine them into one run.
- Candidate snapshot `GC0.f00001` has time `0.5035343677323`; configured N5 and
  N7_H first snapshots have time zero. Candidate frame 79 has time
  `39.50578385037`, so equal filenames are not matched production times.

The first raw file record has global element ID 13142 in the inspected N5, N7,
and N7_H first snapshots. Reading only this record's coordinates gives:

| Case | Native nodes per edge | x interval | y interval | z interval |
| --- | ---: | --- | --- | --- |
| N5 | 6 | `[-6.375,-6.25]` | `[0,.125]` | `[.5,.6815999746322632]` |
| N7 | 8 | `[-6.375,-6.25]` | `[0,.125]` | `[.5,.6815999746322632]` |
| N7_H | 8 | `[-6.375,-6.25]` | `[0,.125]` | `[.5,.6693999767303467]` |

The first two upper z values occur in the historical mesh boundary list; the
N7_H upper value does not. This is a concrete sampled geometry distinction,
not a complete mesh comparison or proof of N5/N7 producer identity. **LIKELY:**
the candidate relates to an earlier channel/legacy mesh family. **UNVERIFIED:**
its connection to any configured production run.

The tar contains 15 members and no solver core tree or Git metadata in its
member inventory. Its archived makefile records a different case path and an
August 2025 generation date. Six small text members were fingerprinted in memory;
the archive was not extracted or rewritten.

## Other local cases are separate evidence

`/data/Nek5000_data/postproc/SIZE:13` sets six nodes. Its `GC.par` requests one
step. `GC.usr:150` loads an existing field, line 153 actively calls `lambda2`,
and line 154 comments out `compute_swirling`. The variable name
`lambdaci_field` therefore does not establish that this active workflow computes
lambda_ci. This is postprocessing code, not an identified production generator.

The practice `GC_small/makefile:5` points to `/home/dai_lab/Nek5000` and requests
eight-byte default real flags. Its log reports 408 elements and five GLL nodes
at lines 66/67. It proves a local practice build configuration, not a production
configuration. The separate `GC_post_lci/GC_small.usr` defines the filtered
swirling criterion described in the [numerical-method page](vortex_numerical_method.md#projected-gradient-swirling-strength).

## Production snapshots and postprocessing metadata

All 668 configured raw headers were inspected without loading velocity fields:

| Case | Snapshots | Nodes per edge | Elements | Stored time extent |
| --- | ---: | ---: | ---: | --- |
| N5 | 81 | 6 | 26,112 | 0 to 20 |
| N7 | 81 | 8 | 26,112 | 0 to 20 |
| N9 | 81 | 10 | 26,112 | 0 to 20 |
| N11 | 101 | 12 | 26,112 | 0 to 50 |
| GC8950_N7 | 81 | 8 | 52,224 | 0 to 20 |
| N7_H | 81 | 8 | 26,112 | 0 to 20 |
| N7_VH | 81 | 8 | 64,600 | 0 to 20 |
| N7_VVH | 81 | 8 | 129,360 | 0 to 20 |

Each series has strictly increasing times and four-byte stored fields. These
observations establish representation/count/time facts, not executable
identity, complete physical geometry, or boundary connectivity.

Existing N7_H frame-79 VTU sidecar metadata records actual time
`19.50240408807`, full native counts `(272,12,8)`, native shape `(8,8,8)`, periodic
axes `(x,y)`, GLL10 sampling, and positive-x output bounds. Its path and digest
are pinned in the evidence inventory. Those records identify postprocessing
configuration and source snapshots; they do not prove original solver BCs or
build flags. Their scientific fields/checkpoints were not modified or rerun.

## Evidence needed to close the gap

For each production case, obtain the matching source checkout/archive with
revision or content identity, user source, actual `SIZE` and parameters, mesh
and BC/connectivity files, build command/transcript, compiler/MPI/library
versions, executable hash, and run log/scheduler command binding that executable
to the snapshots. Resolve runtime/user hooks that alter coordinates, numbering,
or BCs. Confirm storage precision separately from computation precision.

If the original source remains unavailable, Phase 1B may validate against the
pinned local source as a named reference implementation. Such results must
retain the unresolved production-provenance qualification and cannot establish
exact original-solver equivalence.
