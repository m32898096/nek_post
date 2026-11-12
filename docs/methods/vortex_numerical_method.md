# Vortex derivatives, mass recovery, and reconstruction

This page describes the implementation at baseline
`c63bbd8b7b4e1c823ab7336ff07a7c153676d30d`, inspected on 2026-11-12.
The [provenance record](nek5000_provenance.md) defines the limits of the local
Nek5000 comparison; the [Phase 1B specification](phase1b_validation.md) defines
independent validation still needed. File fingerprints and checked source lines
are in [nek5000_evidence.json](nek5000_evidence.json).

## Notation and source roots

Use `p` for polynomial degree and `n=p+1` for the number of GLL nodes per edge.
P7 therefore has eight native nodes. Reference coordinates are
`xi=(r,s,t)`; physical coordinates are `x=(x,y,z)`; velocity is `u=(u,v,w)`.
Native scalar arrays use `(nt,ns,nr)`, so reference directions `(r,s,t)` occupy
NumPy axes `(2,1,0)`. Physical axes can be permuted or reversed relative to these
array axes.

In source citations, `F/` means `/home/dai_lab/Nek5000/core/`, and `P/` means
`src/nek_post/` in this repository. Fortran citations identify the inspected
local files, not upstream master or a proven production revision. Python links
use repository-relative paths and baseline source-line anchors.

## GLL quadrature and differentiation

**CONFIRMED mathematical correspondence.** For `p>=1`, the GLL nodes are the
endpoints `-1,+1` and the interior roots of the derivative of Legendre `P_p`.
The one-dimensional quadrature weights are

\[
w_i=\frac{2}{p(p+1)P_p(\xi_i)^2}.
\]

This quadrature integrates polynomials through degree `2p-1 = 2n-3` exactly
in exact arithmetic. Quadrature weights are distinct from barycentric
interpolation weights.

`F/speclib.f:107` (`ZWGLL`) calls the Jacobi rule with alpha=beta=0;
`F/speclib.f:238` (`ZWGLJD`) constructs the Lobatto rule.
[P/gll.py:20](../../src/nek_post/gll.py#L20) uses Jacobi roots with parameters
`(1,1)` for the interior nodes and
[P/gll.py:50](../../src/nek_post/gll.py#L50) evaluates the weight formula above.
These algorithms implement the same rule without promising identical bits.

For Lagrange basis functions `ell_j`, define `D_ij=ell'_j(xi_i)`. The local
Fortran `DGLL` formula (`F/speclib.f:800`) is

\[
D_{ij}=\frac{P_p(\xi_i)}{P_p(\xi_j)(\xi_i-\xi_j)},\quad i\ne j,
\qquad
D_{00}=-\frac{p(p+1)}4,\quad D_{pp}=\frac{p(p+1)}4,
\]

with zero interior diagonal entries in exact arithmetic. Python builds the
same nodal derivative operator from barycentric basis derivatives, setting each
diagonal by the negative off-diagonal row sum:
[P/gll.py:102](../../src/nek_post/gll.py#L102) and
[P/gll.py:152](../../src/nek_post/gll.py#L152). Node generation and diagonal
rounding differ, so equivalence is mathematical rather than bitwise.

For an element scalar `q[k,j,i]`, the tensor contractions are

\[
q_r[k,j,i]=\sum_l D^{(r)}_{il}q[k,j,l],\quad
q_s[k,j,i]=\sum_l D^{(s)}_{jl}q[k,l,i],\quad
q_t[k,j,i]=\sum_l D^{(t)}_{kl}q[l,j,i].
\]

`F/navier5.f:428` (`local_grad3`) uses matrix products for these contractions.
[P/spectral_derivatives.py:63](../../src/nek_post/spectral_derivatives.py#L63)
uses one-dimensional operators and tensor contractions, including independently
sized local axes. The inspected Fortran routine uses a single `N` for its three
directions. Agreement claims here concern their common supported setting.

## Geometry and all nine physical derivatives

Define the element Jacobian with physical rows and reference columns:

\[
J_{\alpha\beta}=\frac{\partial x_\alpha}{\partial\xi_\beta},
\qquad
\nabla_x q=\nabla_\xi q\,J^{-1}.
\]

For column vectors, Python solves `J^T g_x = g_xi`; it does not need to form an
explicit inverse. The geometry and solve are implemented at
[P/spectral_derivatives.py:117](../../src/nek_post/spectral_derivatives.py#L117)
and [line 134](../../src/nek_post/spectral_derivatives.py#L134).

The velocity tensor convention is

\[
A=\begin{pmatrix}
u_x&u_y&u_z\\
v_x&v_y&v_z\\
w_x&w_y&w_z
\end{pmatrix},\qquad A_{ij}=\partial u_i/\partial x_j.
\]

[P/spectral_derivatives.py:177](../../src/nek_post/spectral_derivatives.py#L177)
solves for all nine components. `F/postpro.f:252` (`comp_gije`) uses the same
component/coordinate convention; its comment contains `du_j` in the denominator,
but the assignments explicitly implement physical x/y/z derivatives.
`F/navier5.f:462` (`gradm1`) provides the scalar-gradient correspondence.

Nek builds coordinate derivatives and metric cofactors in `F/coef.f:488`, maps
geometry metrics to mesh 1 starting at `F/coef.f:532`, and constructs `jacmi`
as the reciprocal Jacobian at `F/coef.f:551`. For example,

\[
q_x=\mathtt{jacmi}\,(q_r\,\mathtt{rxm1}
                           +q_s\,\mathtt{sxm1}
                           +q_t\,\mathtt{txm1}).
\]

The solve and cofactor formulas represent the same transformation when given
the same geometry and operators. **UNVERIFIED production equivalence:** Python
reconstructs metrics from stored snapshot coordinates, while Nek uses internal
geometry and its geometry-mesh path. Four-byte snapshot coordinates and
different arithmetic/summation orders can change results. The precision of the
original production executable cannot be inferred from a snapshot word size.

## GLL lumped-mass recovery

This is the central assembly operation. For local node `a` of element `e`, let

\[
m_{e,a}=|\det J_{e,a}|\,w^{(t)}_{a_t}w^{(s)}_{a_s}w^{(r)}_{a_r}.
\]

Python computes these masses from nodal geometry in
[P/l2_projection.py:230](../../src/nek_post/l2_projection.py#L230), not from
bounding-box volumes. `F/coef.f:735` constructs `bm1 = jacm1 * w3m1` for the
Cartesian case. On a valid positive-orientation Nek mesh, these are the same
mass definition. Python additionally permits consistently reversed local
orientation and uses the absolute determinant for positive integration mass;
that is not a claim that Nek accepts negatively oriented physical meshes.

Let `g(e,a)=I` identify replicas of one continuous global node, including the
explicitly configured periodic equivalence. Define

\[
M_I=\sum_{g(e,a)=I}m_{e,a},\qquad
b_I=\sum_{g(e,a)=I}m_{e,a}q_{e,a},\qquad
(Pq)_{e,a}=\frac{b_{g(e,a)}}{M_{g(e,a)}}.
\]

Equivalently, with Boolean global-to-local replication matrix `Q` and diagonal
local mass `B`,

\[
P=Q(Q^TBQ)^{-1}Q^TB.
\]

This is the projection minimizing the quadrature-weighted nodal discrepancy.
In exact arithmetic it preserves constants and the weighted integral, is
idempotent, and makes every assembled replica agree. It is generally not an
equal-weight arithmetic average when contributing element masses differ.

The explicit correspondence is:

| Stage | Local Nek5000 | Python |
| --- | --- | --- |
| Local mass | `F/coef.f:260` builds tensor weights; `F/coef.f:735` builds `bm1` | [element_gll_weights:230](../../src/nek_post/l2_projection.py#L230) |
| Assembled denominator | `F/coef.f:1500` (`setinvm`): copy `bm1`, `dssum`, reciprocal to `binvm1` | [build_l2_projection_operator:273](../../src/nek_post/l2_projection.py#L273), scatter at line 299 |
| Numerator | `F/navier5.f:583`: multiply local curl by `bm1` | [_project_scalar_component:376](../../src/nek_post/l2_projection.py#L376), multiplication at line 388 |
| Shared sum | `F/dssum.f:33`, gather-scatter addition at line 79 | `np.add.at(rhs, ids, weighted)` at [line 390](../../src/nek_post/l2_projection.py#L390) |
| Inverse assembled mass | Multiply by `binvm1` at `F/navier5.f:585` | Divide by assembled mass at [line 392](../../src/nek_post/l2_projection.py#L392) |
| Replicas | Nek gather-scatter uses its global topology | Gather shared global results to local layout at [line 397](../../src/nek_post/l2_projection.py#L397) |

**CONFIRMED discrete formula correspondence, conditional on matching masses
and topology.** MPI gather-scatter versus serial structured scatter and
different geometry/arithmetic do not establish bitwise field equality.

This is **GLL quadrature-discrete lumped-mass L2 recovery**. A consistent-mass
continuous L2 projection would assemble integrals of basis-function products,
`M_IJ = integral(phi_I phi_J dV)`, and solve the resulting generally nondiagonal
system with its corresponding right-hand side. That solve is not performed
here. The diagonal formula follows from collocated GLL quadrature, which does
not exactly integrate all degree-`2p` basis products even on an affine element.

## Shared nodes, periodicity, and supported geometry

`F/dssum.f:1` (`setupds`) builds the gather-scatter handle from topology-derived
global labels. `F/navier8.f:3` (`set_vert`) includes periodic boundary handling
and the user numbering hook. Nek topology is more general than Python's map.

[P/l2_projection.py:176](../../src/nek_post/l2_projection.py#L176) validates a
complete axis-aligned affine tensor partition, common local shape and axis
mapping, element orientation, GLL profiles, and complete tiling. It rejects
missing/duplicate cells and checks expected element counts when supplied.
[node_ids:137](../../src/nek_post/l2_projection.py#L137) uses structured indices
and modulo identification in caller-supplied periodic axes. For physical axis
`d`, its global node count is

\[
N_d=E_d(n_d-1)+\begin{cases}0,&d\text{ periodic},\\1,&d\text{ nonperiodic}.
\end{cases}
\]

The caller asserts pure translational periodicity and connected interfaces.
Coordinates alone do not establish those boundary conditions. Rotational
periodicity, intentionally disconnected coincident interfaces, nonconforming
interfaces, and general curved/skew global meshes are outside this recovery
implementation. Local derivative and local-weight routines support broader
geometry, with conditioning and nodal determinant checks; those checks do not
certify injectivity between GLL nodes.

For N7_H, the configured postprocessing assumptions are periodic `(x,y)` and
element counts `(272,12,8)`. These are case-specific, not generic defaults for
arbitrary simulations. Under those assumptions, the unique recovery grid is
`(1904,84,57)`. Visualization can retain both physical periodic endpoints,
giving `(1905,85,57)` without adding independent degrees of freedom. See the
[production-linkage limits](nek5000_provenance.md#production-linkage-status).

## Signed omega_x

The streamwise component is

\[
\omega_x=w_y-v_z=A_{32}-A_{23},\qquad
\omega_x^{\mathrm{recovered}}=P(A_{32}-A_{23}).
\]

Here mathematical indices are one-based. Python uses `A[...,2,1]-A[...,1,2]`.
`F/navier5.f:552` has the same sign; `comp_vort3` then applies
`bm1 -> dssum -> binvm1` to the local curl.

The dedicated [P/vorticity.py:13](../../src/nek_post/vorticity.py#L13) solves
`J c_y=e_y` and `J c_z=e_z` for only the required inverse columns, contracts
them with reference derivatives of `w` and `v`, then invokes scalar recovery
in [line 39](../../src/nek_post/vorticity.py#L39). The combined exporter computes
the full local tensor once and recovers its signed difference at
[P/vortex_3d.py:67](../../src/nek_post/vortex_3d.py#L67).

Linearity gives `P(A32-A23)=P(A32)-P(A23)` in exact arithmetic. Different solve
and summation orders can round differently. This identity does not assert that
projection commutes with differentiating a projected velocity field.

## Projected-gradient swirling strength

All nine components are recovered independently using the same masses and node
map in [P/l2_projection.py:442](../../src/nek_post/l2_projection.py#L442).
Let `G=P(A)`. The Python criterion is

\[
\lambda_{ci}(G)=\max_{k=1,2,3}|\operatorname{Im}\lambda_k(G)|,
\qquad \lambda_{ci}^2=(\lambda_{ci})^2.
\]

[P/swirling_strength.py:38](../../src/nek_post/swirling_strength.py#L38)
uses chunked float64 `numpy.linalg.eigvals`, followed by the maximum absolute
imaginary part. There is no trace subtraction, symmetrization, imaginary-part
threshold, or smoothing. Repeated/defective eigenvalues retain the eigensolver's
floating-point sensitivity. Both `omega_x` and `lambda_ci` have inverse-time
units; squared swirling strength has inverse-time-squared units, subject to the
simulation's nondimensionalization.

The order is explicitly **differentiate -> project nine gradients ->
eigenvalues**: [lambda_ci_workflow.py:205](../../src/nek_post/lambda_ci_workflow.py#L205)
then [line 216](../../src/nek_post/lambda_ci_workflow.py#L216), or
[vortex_3d.py:69](../../src/nek_post/vortex_3d.py#L69) then
[line 84](../../src/nek_post/vortex_3d.py#L84). In general,
`lambda_ci(P(A)) != P(lambda_ci(A))` because the criterion is nonlinear.

No `lambda_ci` implementation was found in the inspected local core. Its
`F/postpro.f:19` implements **lambda2**, the middle eigenvalue of `S^2+Omega^2`,
with subsequent filtering. That is a different vortex criterion.

The separate user routine
`/home/dai_lab/Nek5000/run/LX_practice/GC_post_lci/GC_small.usr:172` computes a
cubic-invariant swirling-strength expression from unprojected element-local
gradients (`comp_gije` at line 204), squares it, and filters that scalar three
times (`wght=1`, `ncut=2`, lines 235 onward). Its caller takes the square root
at line 121. `F/postpro.f:343` defines the called scalar filter. This custom
criterion is not core-native `lambda_ci`, and is not the Python projected-tensor
method. Algorithmic, operation-order, and filtering differences must be assessed
separately; agreement must not be forced by changing either method.

## Dense P7 reconstruction and output selection

For the native projected tensor `G`, define component-wise interpolation `I`
using the original degree-seven element basis. At a target point `(r,s,t)`,

\[
(IG)_{ij}(r,s,t)=\sum_{l,m,n=0}^{7}
\ell_l(r)\ell_m(s)\ell_n(t)G_{n,m,l,ij}.
\]

[P/gll.py:196](../../src/nek_post/gll.py#L196) builds the interpolation matrices;
[P/gll_refinement.py:34](../../src/nek_post/gll_refinement.py#L34) applies them
as tensor products. The 3D reconstruction requires a native `(8,8,8)` element
and target node count at least eight at
[vortex_3d_reconstruction.py:68](../../src/nek_post/vortex_3d_reconstruction.py#L68).

At [reconstruct_element:118](../../src/nek_post/vortex_3d_reconstruction.py#L118),
all nine components are interpolated before the eigensolve:

\[
\omega_x^{\mathrm{dense}}=I(P(A_{32}-A_{23})),\qquad
\lambda_{ci}^{\mathrm{dense}}=\lambda_{ci}(I(P(A))).
\]

Interpolating an already evaluated `lambda_ci` scalar is a different method.
The reconstructed scalar curl is audited against the difference of reconstructed
projected tensor components, with the existing scaled `2e-12` tolerance. The
code retains the prescribed scalar route rather than replacing it with the
alternate route.

Native differentiation and recovery use the full original domain. Cropping
affects output selection after recovery, retaining contributions from neighbors
and periodic partners outside the output region. Geometry reconstruction uses
the original stored rectilinear coordinate profiles; shared output traces are
checked. Linear VTK hex subcells and element-local duplicate interface points
are visualization representations, not new spectral-element solution degrees
of freedom. GLL10/GLL12 sampling of P7 is not a higher-order simulation.

## What is and is not established

| Claim | Phase 1A assessment |
| --- | --- |
| GLL rule and nodal differentiation | CONFIRMED mathematical correspondence to the pinned local source |
| Physical transformation and curl sign | CONFIRMED formula/convention correspondence; algorithms and geometry inputs differ |
| `bm1 -> dssum -> binvm1` | CONFIRMED discrete mass-recovery correspondence, conditional on identical geometry and node equivalence |
| Python projected-gradient eigenvalues and tensor-first reconstruction | CONFIRMED implementation order |
| Custom filtered Fortran criterion differs from Python | CONFIRMED method difference |
| Exact production source, executable, flags, and N7_H boundary linkage | UNVERIFIED; historical candidate evidence is not a complete chain |
| Bitwise Nek/Python field equality or independent production numerical validation | UNVERIFIED; Phase 1B work remains |

This documentation does not introduce a numerical change or a new numerical
test result. Existing tests provide useful regression contracts; the
[next-phase specification](phase1b_validation.md) separates those contracts from
independent solver validation.
