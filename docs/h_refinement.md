# H-refinement mesh/data inventory

The h-study is configured independently in `config/h_refinement.yaml`. Its case
sequence is a candidate coarse-to-fine ordering, its expected polynomial order
is an assertion to test, and `N7_VVH` is a provisional reference. Directories are
resolved through `ProjectPaths` and `config/paths.yaml`. The p-study's
`config/cases.yaml`, `orders`, N11 reference, comparison sets, and output roots
are unchanged.

Run from the repository root:

```bash
PYENV_VERSION=research312 python scripts/27_inventory_h_refinement.py
# Optional report capture outside the existing p-study output tree:
PYENV_VERSION=research312 python scripts/27_inventory_h_refinement.py > /tmp/nek_h_inventory.json
```

The command prints JSON and creates no files itself. `--paths-config` and
`--study-config` select alternative configuration files. Scientific inventory
and validation live in `nek_post.h_refinement`; the script handles arguments
and serialization only. A false validation flag is a successful inventory
result, not a CLI error; missing/unreadable or unsupported data exits with an
error.

The inventory reuses exact `PREFIX.fNNNNN` discovery and the project's Nek5000
reader. A header-only adapter in `io_nek` delegates to pymech's header reader.
Every field header is scanned for time, GLL dimensions, element count, and
variable availability. One coordinate-bearing field is read per case, skipping
solution-variable I/O, and all its elements contribute to the domain bounds.
Pymech still allocates payload arrays; large cases can require several GB of RAM.
Cases are processed sequentially.

The JSON includes every discovered filename and physical time, first/last
indices, count, time range, output-spacing min/median/max and uniformity,
representative mesh source, scalar and coordinate array shapes, per-axis GLL
node counts, total element-local GLL nodes, polynomial orders, element count,
x/y/z extents, variable codes/counts, and warnings. Variable-count vectors are
ordered as coordinates, velocity, pressure, temperature, passive scalars.
`temperature` names the Nek storage slot; the header alone does not establish
whether it represents physical temperature or concentration.

Polynomial degree is nodes-per-active-axis minus one. Anisotropic degrees are
reported per axis with no single scalar degree; 2D singleton z is excluded from
the degree calculation. Split-file dumps, nonfinite times/coordinates,
missing coordinates, and changing mesh order/count across headers are rejected.
Index gaps and duplicate/reversed physical times are reported; non-increasing
times suppress the representative spacing. Output spacing is not the solver's
internal timestep. Coordinates are assumed stationary over the series.

## Real-data inspection, 2026-09-16

All 243 headers were scanned. Each case has 81 files, `GC0.f00001` through
`GC0.f00081`, covering physical time 0 through 20 without index gaps. Each uses
8 x 8 x 8 GLL nodes (512 element-local nodes), polynomial degree N=7, scalar
array shape `(8, 8, 8)` in `(lz, ly, lx)` order, and coordinate shape
`(3, 8, 8, 8)`. Element counts and orders remain constant across each series.

| Case | Data directory | Elements | Median output spacing | Min / max spacing |
| --- | --- | ---: | ---: | --- |
| N7_H | `/data/Nek5000_data/case_N7_H` | 26,112 | 0.250333060 | 0.245941252 / 0.255182945 |
| N7_VH | `/data/Nek5000_data/case_N7_VH` | 64,600 | 0.249824948 | 0.246823747 / 0.255800202 |
| N7_VVH | `/data/Nek5000_data/case_N7_VVH` | 129,360 | 0.250426757 | 0.247256798 / 0.253355443 |

All representative meshes (`GC0.f00001`) have identical bounds:
`x = [-17, 17]`, `y = [0, 1.5]`, `z = [0, 1]`. The configurable absolute
comparison tolerance is `1e-6`, with relative tolerance zero. All headers carry
`XUPT`: coordinates, three velocity components, pressure, and one temperature
slot; no passive scalar slots. Output times are strictly increasing but
nonuniform and differ between cases; equal file indices must not be treated as
exactly aligned times in future work.

The metadata confirms the same degree and bounding domain, with strictly
increasing element counts H -> VH -> VVH. This is consistent with h-refinement,
and VVH is the unique largest mesh by element count. A fully controlled pure
h-refinement study is not established by this inventory alone: identical
boundary/initial conditions, physical parameters, solver settings, and domain
topology have not been verified. Equal bounds do not prove equal interior
domains, and largest element count does not prove finest spacing everywhere or
convergence. Mesh coordinate invariance over time is assumed. No field,
front, or leading-edge comparisons are implemented.
