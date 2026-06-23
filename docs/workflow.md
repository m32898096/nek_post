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
