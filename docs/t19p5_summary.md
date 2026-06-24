# t19p5 Summary

The `t19p5` comparison set evaluates time-aligned Nek5000 outputs near `t = 19.5`, using `N11` as the reference case.

Configured file indices:

- `N5`: `f00079`
- `N7`: `f00079`
- `N9`: `f00079`
- `N11`: `f00040`

## Relative L2 Errors

Concentration `C`:

- `N5`: 0.2373
- `N7`: 0.2141
- `N9`: 0.1432

Velocity magnitude:

- `N5`: approximately 0.2684
- `N7`: approximately 0.2430
- `N9`: approximately 0.1753

Pressure fluctuation `p_prime = p - mean(p)`:

- `N5`: 0.2639
- `N7`: 0.2487
- `N9`: 0.1519

## Interpretation

The relative L2 errors of concentration, velocity magnitude, and pressure fluctuation all decrease from `N5` to `N9` when compared with the `N11` reference. This indicates that the lower-order solutions progressively approach the high-order reference solution.

Linf errors are more sensitive to local extrema and should not be the primary convergence indicator.

Velocity and pressure vertical banding is currently interpreted as a feature already visible in the raw midspan slice data rather than a post-processing bug. Diagnostics did not identify slab extraction, duplicate projected `(x,z)` points, or speed computation from interpolated velocity components as the primary cause.
