# Phase 1B independent numerical references

The reference tests were run on `feature/nek5000-numerical-validation` after
Phase 1A commit `7d228df73f7c56e48ebe33fb4885bbea69af1d6f`. The scientific
method and production-source qualifications in the
[method definition](vortex_numerical_method.md),
[provenance record](nek5000_provenance.md), and
[validation specification](phase1b_validation.md) apply.

## Reproducible references

[The new test module](../../tests/test_independent_numerical_reference.py)
contains deterministic analytical fields and explicit connectivity fixtures.
It imports production routines solely as subjects under test. Existing test
helpers and existing golden values supply none of its expected arrays.

| Operation | Independent reference | Validation level |
| --- | --- | --- |
| GLL nodes, weights, D | Actual pinned local `speclib.f` calls to `ZWGLL` and `DGLL` | INDEPENDENTLY NUMERICALLY VERIFIED against the selected local reference build |
| GLL rule and D | Legendre companion-matrix roots, monomial moment linear system, monomial Vandermonde derivative system, exact polynomial derivatives and integrals | ANALYTICALLY VERIFIED with independently evaluated numerical oracles |
| Physical gradients and Jacobian | Explicit affine coordinate maps and symbolic derivatives of quadratic physical velocity fields | ANALYTICALLY VERIFIED for the tested geometries |
| Recovery | Declared vertex/interior connectivity entities and contributor lists, rational P2 weights, volume/8 Jacobians; P7 moment-solved weights | ANALYTICALLY VERIFIED against explicit independent assembly |
| Signed curl | Differentiated translation, shear, rotation, and polynomial fields followed by contributor-list recovery | ANALYTICALLY VERIFIED |
| Eigenvalues | Characteristic quadratics of triangular/block tensors, cyclic component permutations, rotation plus extension | ANALYTICALLY VERIFIED, including small complex pairs |
| Dense reconstruction | Degree-seven polynomial evaluation and block eigenvalues at physical output coordinates | ANALYTICALLY VERIFIED for native P7, GLL10 and GLL12 |
| SBP, constants, conservation, idempotence, residual orthogonality | Algebraic identities of the discrete operators, supplementary to independent array comparisons | SELF-CONSISTENCY ONLY when considered alone |
| Original production solver field equality | Matching source/build/executable/mesh chain remains unavailable | UNVERIFIED |

The mathematical node oracle obtains roots from NumPy's Legendre companion
matrix. The weight oracle solves monomial moments through degree `n-1`.
Independent exact moments through degree `2n-3` additionally test the production
rule. The derivative oracle solves `D V = V'` in a monomial basis. Production
uses SciPy Jacobi roots, the Legendre weight formula, and barycentric
coefficients. Orders are P1, P2, P3, P5, P7, P9 and P11.

### Executed local Fortran reference

[The standalone driver](../../tests/reference/nek_gll_driver.f90) compiles the
unchanged `/home/dai_lab/Nek5000/core/speclib.f`. Its SHA-256 is checked against
[the Phase 1A inventory](nek5000_evidence.json) before compilation. Source
interfaces were inspected at `speclib.f` lines 107, 207, 238 and 800.
`ZWGLL` passes through `ZWGLJ` to double-precision `ZWGLJD`; `DGLL` accepts the
leading dimension separately from the number of nodes.

The executed compiler was GNU Fortran 13.3.0, Ubuntu
`13.3.0-6ubuntu2~24.04.1`. The test passes these flags explicitly.

```text
-std=legacy -fallow-argument-mismatch -fdefault-real-8 -fdefault-double-8
-ffunction-sections -fdata-sections -Wl,--gc-sections
```

Default `REAL` and explicit double precision both occupy eight bytes for this
selected reference build. Garbage collection removes unused spectral routines
and their unresolved dependencies. The driver's `exitt` dependency terminates
with failure if reached. Only GLL routines execute. Compilation and execution
occur in pytest temporary directories. On machines lacking either the local
source or `gfortran`, seven Fortran comparisons explicitly skip. A changed
available source fingerprint fails the test.

`comp_vort3`, `setinvm`, and `dssum` depend on solver include/common state and
Nek topology/gather-scatter setup. This harness executes their GLL dependency
and independently tests their documented mass/curl formulas in Python. Direct
execution of those three solver routines remains UNVERIFIED. In particular,
no solver MPI assembly or production executable was run. The actual local
`navier5.f` lines 552 and 583–585 form signed curl and apply local `bm1`,
`dssum`, then assembled inverse `binvm1`. `coef.f` lines 1500–1530 copy local
mass, assemble, and reciprocate. The scalar and nine-component Python kernel
at `l2_projection.py` lines 376–397 multiplies local mass, accumulates by global
ID, divides by assembled mass and gathers replicas. `comp_gije` produces
local gradients before this recovery. The custom filtered user routine
`GC_post_lci/GC_small.usr` remains a separate method.

## Geometry and connectivity coverage

The gradient tests use all six array/physical-axis permutations, both local
orientation settings, a nonuniform diagonal map and a skew map. All nine
components, the full Jacobian, reference derivatives, physical derivatives,
constant fields and polynomial curl are checked against analytical values.
Skew elements exercise local differentiation.

The recovery fixture declares eight connected boxes with unequal widths in
all three axes, shuffles element order, and labels vertices and element
interiors explicitly. Opposite boundary vertices are explicitly identified for
translation-periodic fixtures. Production integer IDs are compared by their
entire equivalence partition, allowing arbitrary ID numbering. Tests cover
all six axis permutations, mixed local reversals, no periodicity, x
periodicity, and xyz periodicity. Separate P7 cases cover no periodicity and
xy periodicity. Expected weights use analytical element volumes and independent
quadrature values. Expected contributor means exercise faces, edges and
corners, scalar recovery and nine distinct tensor components.

These fixtures assert physical connectivity and translational boundary
partners. External boundary metadata verification and unsupported rotational
periodicity or disconnected coincident interfaces retain the caller
responsibilities specified in Phase 1A. The tests introduce no topology
rejection requirement. The recovery being tested is the GLL quadrature
lumped-mass projection. A consistent-mass continuous Galerkin solve remains
outside this method.

## Errors and tolerances

The following maxima aggregate the 121 new cases. Relative errors use
`max(abs(actual-expected))/max(abs(expected))` within each comparison, then
aggregate across cases. An identically zero reference reports the absolute
residual instead. Pointwise test acceptance uses `atol + rtol*abs(expected)`.

| Comparison | Maximum absolute error | Maximum relative infinity error |
| --- | ---: | ---: |
| GLL nodes against mathematical oracle | 1.78e-15 | 1.78e-15 |
| GLL weights against moment solve | 5.18e-14 | 1.91e-13 |
| D against Vandermonde solve | 5.41e-12 | 1.22e-13 |
| Exact polynomial derivatives | 1.43e-14 | See per-degree XML properties |
| Exact quadrature moments | 8.89e-16 | Zero moments use absolute residuals |
| GLL nodes against compiled Nek | 1.12e-16 | 1.12e-16 |
| GLL weights against compiled Nek | 6.67e-16 | 2.25e-15 |
| D against compiled Nek | 7.11e-14 | 1.60e-15 |
| SBP boundary identity | 4.10e-15 | 4.10e-15 |
| Jacobian | 3.56e-15 | 2.37e-15 |
| Reference gradients | 1.16e-14 | 1.76e-15 |
| All nine physical gradients | 5.16e-14 | 9.54e-15 |
| P2 local and assembled masses | 5.56e-16 | 1.25e-15 |
| P2 scalar recovery | 1.78e-15 | 2.00e-16 |
| P2 nine-component recovery | 2.14e-14 | 3.05e-16 |
| P7 local mass | 4.13e-16 | 2.10e-14 |
| P7 scalar recovery | 1.89e-14 | 4.70e-15 |
| Constants | 0 | 0 |
| Weighted conservation | 2.67e-15 | 5.50e-16 |
| Idempotence | 1.78e-15 | 2.00e-16 |
| Weighted residual orthogonality | 2.23e-16 | Absolute zero-reference residual |
| Polynomial curl on skew/local maps | 8.89e-15 | 1.78e-15 |
| Recovered curl on unequal periodic boxes | 7.11e-15 | 1.78e-16 |
| Analytical lambda_ci | 4.45e-16 | 2.23e-16 |
| Analytical lambda_ci squared | 1.78e-15 | 4.45e-16 |
| Reconstructed nine-component tensor | 2.36e-14 | 1.86e-14 |
| Reconstructed curl | 1.38e-14 | 4.59e-15 |
| Reconstructed lambda_ci | 5.56e-15 | 3.93e-15 |
| Degree-seven component interpolation | 6.13e-14 | 1.05e-14 |

Bounds in this table are rounded upward. The measured errors and test names
are available as JUnit properties when running the command below.
The moment and Vandermonde solves lose accuracy as the monomial basis becomes
ill-conditioned. Their float64 budgets are 3e-12 for weights and 3e-10 for D
through P11. The compiled local Fortran comparison uses 3e-14 for nodes/weights
and 3e-11 for D, accommodating endpoint derivative cancellation and rounding.
Exact polynomial, mass, gradient and reconstruction comparisons use budgets
between 2e-14 and 5e-12 according to operation count and expected field scale.
These tolerances cover small, well-conditioned synthetic geometries.
They provide no tolerance guarantee for arbitrary distorted production meshes.

Eigenvalue cases include uniform translation, real shear/extension spectra,
rotation about all three axes, isotropic shifts, mixed rotation/strain, a
critical defective block and nearly defective blocks on both sides of the
real/complex transition. Small nonzero strengths down to 1e-12 additionally
require relative accuracy of 3e-13 with absolute allowance 1e-26. Production
imaginary-part thresholds and filtering remain absent.

Opposite element rotations give strength two before recovery and zero at their
shared interface after tensor recovery, explicitly distinguishing
`lambda_ci(P(A))` from `P(lambda_ci(A))`. A dense analytical field has strength
`sqrt(1+x^2)`. Interpolating native scalar strengths produces maximum error
6.4751e-7 at GLL12, while tensor-first reconstruction agrees with the analytical
answer within 3.78e-15. P7 sampling retains the original simulation degree.

## Validation commands and observations

Run from the repository root using temporary pytest output locations.

```bash
PYENV_VERSION=research312 python -B -m pytest -q -p no:cacheprovider \
  tests/test_independent_numerical_reference.py \
  --junitxml=/tmp/phase1b-reference.xml -o junit_family=legacy
PYENV_VERSION=research312 python -B -m pytest -q -p no:cacheprovider
```

The new suite passed 121 cases. The focused existing suite passed 797 cases
covering GLL, spectral derivatives, L2 and scalar recovery, vorticity,
swirling strength, reconstruction, workflow regressions and series workflows.
The full suite passed 2,706 tests in 105.57 seconds. New reference tests
passed 121 tests in 1.68 seconds, including the stricter small-imaginary-part
assertion. Syntax parsing and whitespace checks passed. Git confirms the
Phase 1A HEAD and tracked files are unchanged. Three new files comprise the
uncommitted Phase 1B work.

The initial reconstruction fixture used `(lo+hi)/2 + (hi-lo)/2*q` at shared
endpoint `0.4`. Adjacent elements produced `0.39999999999999997` and `0.4`.
The exporter rejected those unequal coordinates. The fixture now assigns its
declared interface endpoints exactly, retaining the same physical mesh and
analytical reference. The initial test source and failure XML were preserved
in `/tmp/phase1b-initial-reference.py` and `/tmp/phase1b-initial-reference.xml`.
No tolerance was relaxed. This was a fixture discrepancy. No production
numerical discrepancy was confirmed by the completed reference comparisons.

Original production source/build linkage, solver-generated intermediate
metrics/masses, actual solver equivalence labels, compiled curl/gather-scatter
comparisons, and production field equality remain UNVERIFIED. The tests use
bounded synthetic arrays, with files confined to isolated pytest temporary
locations. Scientific datasets and production outputs were untouched.
