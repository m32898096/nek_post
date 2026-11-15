# ADR-0001: Package topology

Status: **DESIGN RECORDED / INDEPENDENT DOCUMENTATION REVIEW PENDING**. This documents the accepted direction; it does not claim implementation acceptance or authorize subsequent work. Source baseline: `3f9be2e4cd088c439700ba857e344adde9025712`.

## Context

The repository already has a setuptools src package, 99 Python package files and 46 script entrypoints. Exact source fingerprints, sibling-file lookup and module-qualified workers make source relocation a compatibility change. The existing package discovery is in pyproject.toml:1–13; the protected inventory is recorded in [acceptance gates](../acceptance_gates.md).

## Alternatives

Keeping only the current scripts minimizes change but leaves installed execution and portable configuration unresolved. Wholesale relocation would improve visual grouping while changing imports, source identities and path guards. Conservative Extension adds an execution boundary around the existing package and can be reviewed capability by capability.

## Decision

Use Conservative Extension. Keep existing scientific APIs, source locations and legacy implementations. Add a future CLI package/module entrypoint, static registry, resolver, destination planner and narrowly justified adapters. Prefer an existing complete workflow API over reconstructing its numerical work. Proposed responsibilities are defined in [target architecture](../target_architecture.md).

## Rationale

The installation, configuration and destination problems can be solved outside the scientific kernels. This keeps each change small enough to compare against established behavior and leaves the original research entrypoints available throughout Phase 4.

## Trade-offs

Some flat modules and duplicate orchestration will remain. Both old and new entrypoints need maintained contracts. A new adapter must justify its maintenance cost; when an abstraction adds little value, defer that route instead of expanding the refactor.

## Compatibility consequences

Preserve all 46 original implementations, complete protected bytes, sibling lookup and worker/plan qualified names. Additional unsigned files do not establish behavioral compatibility. Native defaults and guards still apply unless the selected API supports explicit parameters; do not replace protected files with facades.

## Validation requirements

Require G-FREEZE, G-LEGACY and G-HELP, plus the numerical and filesystem gates for each capability. Compare installed behavior from outside the checkout. Review each complete batch diff and its evidence independently. These runtime tests have not been executed during this documentation task.

## Deferred decisions

Domain-wide package regrouping, public API redesign and broad test relocation are outside this CLI foundation. Any future signed-source change needs its own version/provenance decision; this ADR does not authorize it.
