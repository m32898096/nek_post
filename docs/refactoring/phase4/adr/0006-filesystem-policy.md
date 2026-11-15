# ADR-0006: Destination translation and native filesystem lifecycles

Status: **DESIGN RECORDED / INDEPENDENT DOCUMENTATION REVIEW PENDING**. This documents the accepted direction; it does not claim implementation acceptance or authorize subsequent work. Source baseline: `3f9be2e4cd088c439700ba857e344adde9025712`.

## Context

Delegates interpret output roots differently. Lambda slices write adjacent slices and case PVDs; omega appends the case; vortex conditionally adds gllN. Cache replacement can create and delete siblings. Native locking, collision and publication policies also differ.

## Alternatives

Passing one universal directory directly to every delegate risks incorrect nesting or sibling writes. Unifying scientific writers would change protected bytes and lifecycle rules. Route-specific translation with complete write planning can preserve those rules and reject unknown writes.

## Decision

Use the [per-command destination matrix](../destination_contracts.md). Every writing route requires explicit R and a mandatory [Filesystem Safety Context](../configuration_dependencies.md#mandatory-filesystem-safety-context) from --paths-config or NEK_POST_PATHS_CONFIG, with no implicit machine fallback, independent of scientific config. Resolve/validate all configured raw cases and original historical/input/reference/package roots before allocation or write-capable handler imports; incomplete context refuses execution. It plans final products, parent creation, companions, caches, temporaries, backups, locks and deletions. Preserve native guards and locked revalidation. Help, plan and validation-only routes allocate no operational directories.

## Rationale

A route-specific translator can keep the native write closure inside authorization without editing protected modules. Supplementary CLI checks and native guards both remain authoritative; an authorized root is not permission to bypass an existing refusal.

## Trade-offs

Some full transitive write sets remain Unknown and must be closed before route acceptance. Direct writers can leave partial output. Cooperative native locks do not defeat hostile concurrent filesystem replacement. No universal transaction, atomic publication or dry-run purity is promised.

## Compatibility consequences

R bounds ordinary scientific, operational and preview-creation writes. Navigation preview creation writes its complete output tree to authorized V beneath R. Publication reads reviewed V as a read-only input and may exclusively create only P/README.md and P/results_index.csv under separately explicitly authorized P. Existing publication targets must never be replaced. A partial pair remains and blocks retries. No other writes outside R are authorized by the current architecture. Any future proposal for preview creation outside R requires a separate architecture decision, explicit authorization model and independent review. Explicit eight-node vortex sampling does not add gll8. Preserve native same-directory staging and control cache environment before imports. Benchmark extraction --output-dir D is an exact final directory under the parent R, without appended case/frame components; warmups and measured runs share that contract. P7 series uses its persistent series-wide lock and private PVD temporary/hard-link no-replace publication; no additional per-PVD lock is asserted.

## Validation requirements

Require G-WRITE, G-PLAN, G-LIFECYCLE and G-RESULTS. Observe every enabled leaf/flag branch, including deletion, failure, cache and external-tool paths. Verify missing/invalid/incomplete safety context and selected/unselected raw-case, historical P/H, symlink/traversal/input overlap refusals before any allocation/import side effects; verify valid R, native collisions/locks, exact benchmark D and the narrow P exception. No writer was executed in this task.

## Deferred decisions

Hostile-race hardening, universal transactions, new resume protocols and historical-result migration need separate decisions. Unknown write sets require source and observed evidence; changing a documentation label does not close the gate.
