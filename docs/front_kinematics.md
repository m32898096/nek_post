# Front Kinematics

The front-position diagnostic reads each case's `front_simple.dat` file to obtain `x_front(t)`.

The script computes front velocity by differentiating `x_front(t)`, smooths the velocity, and then integrates the smoothed velocity back in time to reconstruct `x_front(t)`. The reconstruction error checks whether the smoothing preserves the original front trajectory.

The slumping-region velocity is estimated by a linear fit of front position over:

```text
3 <= t <= 12
```

Both the original front position and the reconstructed front position are fit over that window when enough samples are available.

Run the diagnostic with:

```bash
python scripts/13_front_kinematics.py \
  --cases N5,N7,N9 \
  --overwrite
```

By default, inputs are read from:

```text
/data/Nek5000_data/case_N5/front_simple.dat
/data/Nek5000_data/case_N7/front_simple.dat
/data/Nek5000_data/case_N9/front_simple.dat
```

Outputs are written under:

```text
/data/Nek5000_data/results/poly_order_compare/front_kinematics
```
