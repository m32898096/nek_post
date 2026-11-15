# ADR-0008: Retained scientific feature branches

Status: **DESIGN RECORDED / INDEPENDENT DOCUMENTATION REVIEW PENDING**. This documents the accepted direction; it does not claim implementation acceptance or authorize subsequent work. Source baseline: `3f9be2e4cd088c439700ba857e344adde9025712`.

## Context

The baseline retains feature/energy-budget-closure and feature/n7-spectral-volume-export. Their additional energy definitions and uniform-volume VTI capability require separate semantic and scientific assessment. Existing dev energy and VTU workflows can receive CLI adapters without integrating those branches.

## Alternatives

Automatic integration makes CLI acceptance depend on additional numerical changes. Deletion would discard unresolved research work without authorization. Retaining the references and reviewing capabilities separately preserves both the foundation scope and the outstanding research decisions.

## Decision

Do not merge or delete either retained branch. Do not advertise its additional definitions or VTI capability as accepted dev CLI behavior. Energy remains within existing P-refinement scope, and vortex routes describe existing native/reconstructed VTU contracts. See [target architecture](../target_architecture.md).

## Rationale

The CLI task establishes execution, configuration and destination boundaries. Separate capability decisions prevent documentation or installation acceptance from being mistaken for scientific approval of additional methods.

## Trade-offs

Retained branches can diverge and need future ownership and review. This ADR does not assess their full diffs or merge conflicts. Deferral neither validates nor dismisses their scientific content.

## Compatibility consequences

Preserve dev, master and both retained branch SHAs. Only the separately authorized documentation branch is created here. VTI sampling/representation is not established by VTU tests, and additional energy definitions are not established by existing P-energy tests.

## Validation requirements

A separately authorized task must inspect branch diffs, definitions, units, operation order, sampling/serialization, independent references and provenance before integration. Merge or cleanup then requires its own authorization. This task verified reference preservation and did not test branch capabilities.

## Deferred decisions

Final integration, retention or deletion, a possible VTI command hierarchy and additional energy capabilities remain separate decisions. Their unresolved disposition does not block the existing dev CLI foundation.

## Verified starting refs

| Branch | SHA |
| --- | --- |
| dev | `3f9be2e4cd088c439700ba857e344adde9025712` |
| master | `72f08bdb64dd45785450189da677031f44b15439` |
| feature/energy-budget-closure | `9ca8102e14d74550ee435d8a561d4356a6be86a5` |
| feature/n7-spectral-volume-export | `7251d2c59d58fe2f42109765b2ccbf2625e20667` |

來源是本次 preflight read-only git show-ref --heads；不是 remote-current 宣稱。新 documentation branch 保持同一 accepted HEAD，未 commit。
