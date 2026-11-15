# ADR-0005: Configuration, resources and dependency capabilities

Status: **DESIGN RECORDED / INDEPENDENT DOCUMENTATION REVIEW PENDING**. This documents the accepted direction; it does not claim implementation acceptance or authorize subsequent work. Source baseline: `3f9be2e4cd088c439700ba857e344adde9025712`.

## Context

The current loader reads YAML, and ProjectPaths converts known values to Path without the proposed portability rules. Legacy defaults depend on checkout configuration and machine paths. Plotting imports and external prerequisites vary by capability; one universal dependency or backend guarantee would be inaccurate.

## Alternatives

Installing machine paths as defaults risks incorrect reads and writes elsewhere. Deep merging obscures override identity. Resolving every string would corrupt scientific labels and methods. Whole-file overrides, field-specific path resolution and per-command capability checks give explicit behavior.

## Decision

Adopt [configuration/dependency contracts](../configuration_dependencies.md): explicit file, corresponding environment variable, packaged cases/study default, then an actionable missing-machine-path error. Invalid overrides never fall back. Resolve YAML path values against the canonical supplying file parent; direct paths use invocation cwd. Separate resolved input/output views and package scientific resources. Every writing invocation, including installed dry-run and explicit-path-only scientific routes, must resolve --paths-config or NEK_POST_PATHS_CONFIG as a mandatory Filesystem Safety Context with no implicit machine fallback. Inventory all configured raw cases, original historical roots including derived H root, explicit inputs/references and checkout/installed/native protected ancestors before any operational allocation, effective YAML or write-capable handler import; incomplete/unsafe context keeps the route unavailable. Roles/order and read-only exemptions are defined in the linked contract.

## Rationale

This preserves typed scientific identity while allowing execution outside the checkout. External machine configuration stays user-controlled. Selected prerequisite checks can fail before work allocation, and static help can remain available without loading scientific handlers.

## Trade-offs

Whole-file overrides require a complete required schema. Symlinked configuration follows the target file's parent, which must be documented. Dependency profiles need branch-specific verification. CLI process isolation bounds backend effects; in-process programmatic calls can still change backend state.

## Compatibility consequences

Keep legacy configuration behavior and the raw paths.yaml schema. Child effective paths YAML preserves normalized original input/safety roots; R-derived output translation stays in a separate adapter view/record so historical roots cannot disappear from safety checks. Explicitly populate the H output root instead of relying on its data_root fallback. Preserve labels, Reynolds numbers, methods, indices and tolerances. Absolute paths go only into APIs that support them; benchmark case remains a label.

## Validation requirements

Require G-CONFIG, G-HELP, G-PLATFORM and G-WRITE. Cover precedence, invalid overrides, symlinks, resource inclusion in wheel/sdist, effective-YAML semantic/safety identity; missing/invalid/incomplete safety source, selected/unselected raw-case and historical P/H overlap, symlink/input overlap, zero allocation/import side effects on refusal, valid R/narrow P and zero-write help/plan/read-only; cache environment before imports, P7 single-frame Poppler, diagnostic PyVista and Linux publication prerequisites. No build/install/runtime tests were run here.

## Deferred decisions

Deep merging, user profiles, resolver public API, semantic-hash encoding, dependency version ranges and non-Linux support remain future decisions. They cannot introduce machine operational defaults or relax native resume checks.
