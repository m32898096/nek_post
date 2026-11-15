# ADR-0007: Results and provenance boundary

Status: **DESIGN RECORDED / INDEPENDENT DOCUMENTATION REVIEW PENDING**. This documents the accepted direction; it does not claim implementation acceptance or authorize subsequent work. Source baseline: `3f9be2e4cd088c439700ba857e344adde9025712`.

## Context

Historical catalog bundles have incomplete producer/revision provenance. inventory_only does not establish scientific validation. Native metadata, manifests, PVDs and artifact hashes define their own completion and reuse conditions. New execution records must not replace those conditions.

## Alternatives

Rewriting historical results into one schema exceeds scope and risks provenance. Recording only an exit code omits effective inputs/configuration/environment. Designing final results_v2 now couples the CLI foundation to a separate migration. Keep native artifacts and add separate operational records.

## Decision

Keep raw snapshots, historical scientific outputs and native manifests immutable. New runs record execution identity under R/.cli while native writers keep their own metadata. Separate execution completion, artifact validation and scientific accuracy. Preserve four distinct results operations; defer final results_v2 to Phase 4E.

## Rationale

Native validators encode workflow-specific scientific and filesystem conditions. Invocation records explain how a command was launched without rewriting that judgment. Keeping provisional inventory classifications avoids treating names, catalog grouping or process exit as scientific proof.

## Trade-offs

Operational and scientific metadata coexist and need clear interpretation. Historical producer gaps remain unknown. Catalog identity depends on relative paths, so moving datasets can change identity; this task does not move them. Metadata inventory cannot authenticate field payloads or solver history.

## Compatibility consequences

build_catalog returns memory; write_catalog creates a fresh external tree; prepare_navigation writes a preview. publish_navigation compares the fresh inventory with reviewed V and exclusively creates two P files, preserving partial failure. See results_catalog_publication.py:57–100 and [destination contracts](../destination_contracts.md). Do not inject new invocation hashes into protected manifests.

## Validation requirements

Require G-RESULTS, G-WRITE and native lifecycle validation. Cover no-write inventory, fresh catalog/preview, stale V, collisions including dangling symlinks, partial publication and retry refusal. Preserve native schemas and artifact bytes. All new runtime checks remain unexecuted.

## Deferred decisions

Final results_v2, historical producer/revision backfill, catalog identity migration and archive authentication belong to Phase 4E or separate tasks. The operational record schema still needs implementation review within the required identity and validation boundaries.
