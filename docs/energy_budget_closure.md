# Energy Budget Closure

The teacher-provided `energy_budget.dat` files contain five columns:

```text
time E_k E_p E_total epsilon
```

Here `epsilon` is treated as the dissipation rate. The main closure check is:

```text
E_total(t) + integral epsilon dt ~= constant
```

For the current Re3450 cases, the expected constant is approximately `12`. The script also computes a pointwise differential check:

```text
dE_total/dt + epsilon ~= 0
```

If `energy_closure = E_total + integral epsilon dt` is nearly constant but offset from `12`, the budget is internally consistent but may use a different normalization. If `energy_closure` drifts strongly, there may be an energy budget mismatch, sign issue, or integration issue.

Run the check with:

```bash
python scripts/12_check_energy_budget_closure.py \
  --cases N5,N7,N9 \
  --target 12 \
  --overwrite
```

By default, inputs are read from:

```text
/data/Nek5000_data/case_N5/energy_budget.dat
/data/Nek5000_data/case_N7/energy_budget.dat
/data/Nek5000_data/case_N9/energy_budget.dat
```

Outputs are written under:

```text
/data/Nek5000_data/results/poly_order_compare/energy_budget_closure
```
