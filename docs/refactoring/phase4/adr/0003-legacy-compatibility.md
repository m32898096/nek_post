# ADR-0003: Legacy entrypoint preservation

Status: **DESIGN RECORDED / INDEPENDENT DOCUMENTATION REVIEW PENDING**. This documents the accepted direction; it does not claim implementation acceptance or authorize subsequent work. Source baseline: `3f9be2e4cd088c439700ba857e344adde9025712`.

## Context

The 46 script paths, globals and helpers have research and file-based test consumers. Script 03 has three field branches and mixed NPZ/table/log behavior. Script 07 has deduplication and stage-order contracts. Script 37 historically publishes a PVD despite its validate-oriented name.

## Alternatives

Removing old paths breaks established commands and historical references. Replacing them with re-export wrappers can change implementation globals, monkeypatch effects, argparse behavior and __file__ context. Keeping original implementations while adding explicit-context adapters provides a stable comparison boundary.

## Decision

Retain all 46 original script implementations and paths through Phase 4. Add the [49-leaf mapping](../cli_migration_matrix.md) separately. Characterize scripts 03 and 07 in B3 before migration. Preserve aliases, globals, parser conventions, side effects, schemas and workflow-specific skip/overwrite/error rules.

## Rationale

A retained implementation provides an observable baseline and a narrow rollback boundary. New CLI configuration and authorization can be explicit without silently changing existing research commands. Complete package workflows can be reused; script-local orchestration requires an adapter.

## Trade-offs

Old and new entrypoints require parallel compatibility evidence. A successful help command or unchanged test count is insufficient. New root flags can differ, but scientific selections, calculation order and artifact schemas need explicit comparison.

## Compatibility consequences

Legacy configuration stays checkout-based. Calling an old main is not sufficient for installed portability. Legacy 37 continues publication; new omega validation is read-only. Script 22 remains a separate directional GLL command. External callers and serialized private objects are not completely inventoried, so avoid unnecessary identity changes.

## Validation requirements

Require G-LEGACY for every original path, including globals and actual monkeypatch consumers. Cover script 03 field branches, script 07 ordering/deduplication and benchmark defaults. Compare meaningful isolated arrays, metadata, CSV ordering, skip-before-read and failure effects. These tests were not run here.

## Deferred decisions

Deprecation, wrapper replacement, private ABI redesign and helper relocation need separate decisions. If an adapter cannot preserve behavior within scope, keep that new route unavailable rather than changing the original script.
