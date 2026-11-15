# ADR-0002: CLI dispatch and installed execution

Status: **DESIGN RECORDED / INDEPENDENT DOCUMENTATION REVIEW PENDING**. This documents the accepted direction; it does not claim implementation acceptance or authorize subsequent work. Source baseline: `3f9be2e4cd088c439700ba857e344adde9025712`.

## Context

Scripts 05/07 and the leading-edge benchmark launch checkout-bound scripts and use repository cwd. See scripts/05_run_t19p5_pipeline.py:44–49, scripts/07_run_multitime_pipeline.py:90–150 and leading_edge_benchmark.py:248–321,874–904. Eager handler imports would also make help depend on scientific and visualization packages.

## Alternatives

Repository-script subprocesses retain checkout behavior but do not provide wheel portability. Eager registration imports optional stacks before help. Direct calls suit complete workflow APIs, while pipelines and benchmarks need explicit child-process boundaries. Static argparse metadata supports help before selected handlers are loaded.

## Decision

Use argparse with a static, standard-library-only registry and lazy dispatch. Use direct APIs or workflow adapters as appropriate. Installed children use sys.executable -B -m nek_post.cli with argument arrays and shell=False. Preserve the original scripts. Installed benchmark children retain parent authorized R in --output-root and receive each native run_directory D via --output-dir as the exact final artifact directory, with no appended case/frame/leading-edge component. D must pass the common safety/containment/collision checks; case remains a separate label. The identical warmup/measured handshake, argv and direct D/L_leading_edge_timeseries.csv plus D/L_leading_edge_metadata.csv locations are specified in [target architecture](../target_architecture.md#exact-benchmark-child-output-handshake-m1).

## Rationale

Module execution uses the parent's installed interpreter. Explicit configuration, inputs, output root, cwd and environment remove checkout assumptions. The process boundary also isolates global plotting state and retains meaningful child-process benchmark measurement.

## Trade-offs

The registry must explicitly maintain all 49 leaves. Child startup remains part of benchmark measurement. New adapters must preserve script-local ordering and IO behavior. Unimplemented commands can be discoverable but must report unavailable on execution.

## Compatibility consequences

Pipeline 05 retains extraction followed by field comparison/plotting. Pipeline 07 retains first-occurrence case/index deduplication and ordered set/field stages. Benchmark None-default hooks are proposed, not existing APIs: preserve late lookup, LC_ALL=C, monkeypatch behavior and case labels. Never pass a pathname into its uppercased case field.

## Validation requirements

Require G-HELP, G-PLAN, G-EXEC and G-LEGACY. Test checkout-free wheel children, effective configuration identity, stdout/stderr separation, first failure, signal mapping and process-group cleanup. C7 depends on C4/C5 and applicable reporting; C11 must first validate exact contained output override/no nesting/outside-R refusal; C12 depends on C11 and reviewed benchmark hooks, then validates checkout-free argv, exact warmup/measured CSV placement/readback and unchanged schemas/equality/legacy defaults. Tests remain unexecuted.

## Deferred decisions

Exact handler filenames, registry representation, interruption escalation timings and non-POSIX support remain implementation decisions. They cannot weaken zero-write help/plan, failure preservation or native staging retention.
