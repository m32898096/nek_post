# Workflow

This document will describe the full Nek5000 post-processing workflow.

Planned stages:

1. Check expected input files.
2. Inspect a single Nek5000 file.
3. Extract a midspan `y`-slice.
4. Interpolate onto a common `x-z` grid.
5. Compare `N5`, `N7`, and `N9` against `N11`.
6. Generate summary plots and tables.

## Extract One Midspan Slice

Run the slice extractor from the repository root:

```bash
python scripts/02_extract_midspan_slice.py --case N11 --index 80
```

The script reads one Nek5000 file, extracts the configured `y`-midspan slab, and saves a compressed `.npz` file under `/data/Nek5000_data/postproc/poly_order_compare/slices/<case>/`. Existing slice files are preserved unless `--overwrite` is passed.

## Compare Polynomial Orders

Nek5000 file indices may not correspond to the same physical time across polynomial-order cases. Compare time-aligned slice files, not necessarily identical file indices. The configured `t19p5` comparison set uses `N5/N7/N9 f00079` and `N11 f00040`:

```bash
python scripts/03_compare_poly_orders.py --comparison-set t19p5
```

If a slice is missing, generate only that slice first, for example:

```bash
python scripts/02_extract_midspan_slice.py --case N11 --index 40 --overwrite
```

The comparison script reads only slice `.npz` files, reports each case/index/time combination, warns when the time mismatch exceeds `--time-tolerance`, and can fail on mismatch with `--strict-time`. Ad hoc time-aligned comparisons can also use `--case-indices N5=80,N7=80,N9=80,N11=40`.

For quick checks where all cases intentionally use the same file index, the old shorthand still works:

```bash
python scripts/03_compare_poly_orders.py --index 80
```

The script interpolates concentration onto a common overlapping `x-z` grid and writes CSV tables under `/data/Nek5000_data/results/poly_order_compare/tables/`.
