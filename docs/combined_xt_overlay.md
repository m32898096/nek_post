# Processed x-t Paper Overlay

This diagnostic overlays processed front reconstructions against digitized Cantero et al. Figure 5a Re3450 data.

Inputs:

- processed front kinematics CSV files under `/data/Nek5000_data/results/poly_order_compare/front_kinematics/`
- digitized paper data at `/data/Nek5000_data/cantero/cantero_fig5a_3D_Re3450.csv`

The script reads the `x_reconstructed` column from each processed front-kinematics CSV. It does not read raw `front_simple.dat` files and does not recompute the processed front trajectory. If the processed CSV files are missing, run:

```bash
python scripts/12_analyze_front_kinematics.py \
  --cases N5,N7,N9 \
  --overwrite
```

The paper CSV is interpreted as:

- column 0: time
- column 1: paper front position

For plotting and comparison, processed `x_reconstructed` is shifted by its initial value so the processed curves use the same front-displacement convention as the digitized paper curve.

Run:

```bash
python scripts/14_plot_combined_xt_overlay.py \
  --cases N5,N7,N9 \
  --overwrite
```

Outputs are written under:

```text
/data/Nek5000_data/results/poly_order_compare/combined_xt_overlay
```

Generated files:

- `processed_xt_paper_overlay_linear.png`
- `processed_xt_paper_overlay_loglog.png`
- `processed_xt_paper_overlay_slumping_region.png`
- `processed_xt_paper_overlay_summary.csv`
