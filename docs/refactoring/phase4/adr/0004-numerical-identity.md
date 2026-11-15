# ADR-0004: Numerical and runtime identity

Status: **DESIGN RECORDED / INDEPENDENT DOCUMENTATION REVIEW PENDING**. This documents the accepted direction; it does not claim implementation acceptance or authorize subsequent work. Source baseline: `3f9be2e4cd088c439700ba857e344adde9025712`.

## Context

Five exact source-fingerprint groups cover 25 distinct files. vtu_compat.py has additional self-identity checks. These checks use complete source bytes and sibling files. Resume also checks configuration, absolute paths, software/Python and artifacts; see vortex_3d_series.py:177–204,342–371.

## Alternatives

Numerical tolerance checks alone do not satisfy existing source identity. Rewriting stored hashes after refactoring would invalidate provenance. Retaining full source/runtime contracts permits a CLI extension while keeping installation availability distinct from historical resume eligibility.

## Decision

Freeze all 25 complete protected sources, existing locations and sibling lookup, plus vtu_compat.py identity. Preserve worker/plan qualified names, numerical definitions, operation ordering, native manifests, strict resume/recovery and environment checks. Do not change metadata or weaken validators to make a new environment appear compatible.

## Rationale

Existing artifacts rely on these identities. Keeping source bytes avoids direct signature invalidation; keeping path, environment and artifact checks preserves the native distinction between eligible reuse and incompatible execution. A CLI layer does not require scientific-source edits.

## Trade-offs

The freeze prevents even comment/import cleanup in signed files. Equal bytes still do not prove reproducibility: unsigned consumers, dependencies and filesystem behavior can affect execution. Additional behavioral and numerical evidence remains necessary.

## Compatibility consequences

Keep griddata/SEM, sampled L2/GLL recovery, external/concentration fronts, Cantero/other definitions, rightmost/Moore, native/tensor-first lambda, native/saved-scalar omega and VTU/VTI distinct. Preserve spawn/pickle module identities and actual-time alignment. The complete hashes and source definitions are in [acceptance gates](../acceptance_gates.md).

## Validation requirements

Require G-FREEZE, G-NUMERIC and G-LIFECYCLE. Include independent numerical expectations, eligible no-recompute resume and rejection of changed source/configuration/path/environment/artifacts. Never regenerate production goldens to hide differences. This task checked source identity only; runtime validation was not executed.

## Deferred decisions

Signed-code refactoring, dataset relocation, metadata rebinding, new numerical variants and solver-equivalence claims need separate review. The supported environment matrix remains a future B3/4D gate.
