# Figure 5a Paper Overlay

This diagnostic overlays polynomial-order front positions against digitized Cantero et al. Figure 5a data for the Re3450 case.

The paper CSV is read from:

```text
/data/Nek5000_data/cantero/cantero_fig5a_3D_Re3450.csv
```

The first column is interpreted as time, and the second column is interpreted as front position `x_front - x_0`. The simulation inputs are `front_simple.dat` files for `N5`, `N7`, and `N9`. Their absolute front positions are shifted by each case's initial front position before comparison so the simulation curves are also plotted as `x_front - x_0`.

The script writes linear and log-log overlays. The log-log overlay is the closest to the style of Cantero et al. Figure 5a.

Metrics are computed by interpolating each simulation curve onto the paper time points over the overlapping time range. Slumping-region metrics use:

```text
3 <= t <= 12
```

Run the overlay with:

```bash
python scripts/14_fig5a_paper_overlay.py \
  --cases N5,N7,N9 \
  --overwrite
```

Outputs are written under:

```text
/data/Nek5000_data/results/poly_order_compare/fig5a_paper_overlay
```
