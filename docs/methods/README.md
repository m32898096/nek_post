# Numerical methods and solver traceability

Phase 1A records the implemented vortex method and the available solver
provenance. Evidence was inspected on **2026-11-12**, against repository baseline
`c63bbd8b7b4e1c823ab7336ff07a7c153676d30d`.

| Read first | Purpose |
| --- | --- |
| [Vortex numerical method](vortex_numerical_method.md) | Definitions, operation order, mass recovery, source correspondence, and supported geometry |
| [Nek5000 provenance](nek5000_provenance.md) | Local source identity, historical build/mesh evidence, and unresolved production linkage |
| [Phase 1B validation specification](phase1b_validation.md) | Independent comparisons needed before claiming solver equivalence |
| [Evidence inventory](nek5000_evidence.json) | SHA-256 fingerprints, source-line checks, raw-header observations, and bounded mesh/coordinate evidence |

**CONFIRMED** means directly observed in the identified files, or an algebraic
consequence of the documented discrete operators. **LIKELY** denotes an
explicit inference. **UNVERIFIED** means the necessary evidence or independent
comparison is absent. Source correspondence alone is not an executed numerical
comparison.

The Python method is local GLL differentiation, physical-gradient
transformation, and GLL lumped-mass shared-node recovery. Signed `omega_x`
projects the local curl. `lambda_ci` evaluates eigenvalues of the nine recovered
gradient components. Dense P7 output reconstructs those components before the
eigensolve. These operation orders are preserved.

**Production provenance remains unresolved.** The available local core identifies
as `20.0-dev`, without recoverable Git metadata. A historical channel case adds
build, executable, and boundary evidence, but does not establish the source or
executable used for N7_H. No production numerical defect or solver-generated
reference comparison is established by Phase 1A.

Existing workflow pages remain available:

- [Script index](../script_index.md)
- [Native omega_x and historical validation](../nek5000_vorticity_algorithm.md)
- [Native/P7 3D VTU and sequential series](../vortex_3d_binary_vtu.md)
- [Dense lambda_ci slices](../lambda_ci_highorder_slices.md)
- [Saved omega_x P7 slice series](../omega_x_p7_slice_series.md)

Phase 1A changes documentation only. It does not execute a solver, generate
reference fields, modify tests, or alter scientific outputs/checkpoints.

To recheck the pinned file bytes and 52 source-line assertions on the inspected
machine, run this read-only check from the repository root. Missing or changed
external files are a failed evidence refresh, not permission to update them:

```bash
PYENV_VERSION=research312 PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import hashlib
import json
from pathlib import Path

evidence = json.loads(Path("docs/methods/nek5000_evidence.json").read_text())
for record in evidence["files"]:
    payload = Path(record["path"]).read_bytes()
    assert len(payload) == record["bytes"], record["path"]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"], record["path"]
for citation in evidence["source_citations"]:
    lines = Path(citation["path"]).read_text().splitlines()
    assert citation["contains"].lower() in lines[citation["line"] - 1].lower(), citation
print("Pinned files and source-line assertions match.")
PY
```
