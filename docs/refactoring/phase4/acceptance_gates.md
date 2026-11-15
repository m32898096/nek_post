# Phase 4A：Acceptance gates 與 evidence boundary

## 本次文件工作與未來 implementation

本文件的 G-* 都是 **FUTURE TEST / NOT RUN**，不是已執行測試結果。此次只做文件的 source/AST grounding、coverage/link/structure/whitespace checks，以及 baseline/source/ref/external metadata preservation checks；不執行 pytest、CLI/help runtime、scientific workflows、resume/recovery/publication、package build/install。

Historical 2,800 passed / 136.89 seconds 與 94 provisional inventory_only bundles 由使用者提供，未於本次重現；不能據此宣稱新 CLI 已測試或舊 catalog 已科學驗證。Source-byte equality 只驗證指定 identity，不能推論 environment/behavior eligibility。

## Numerical/source freeze

本次 preflight 五組 fingerprints 全部 MATCH，distinct union **25**；source membership 亦對照下列 implementations。既有 capture 文件的 baseline 是其歷史 snapshot，不要把那個欄位改成新 HEAD；本次 source comparison 仍符合該 capture。

| Group | Exact members | Current signature definition |
| --- | ---:| --- |
| vortex_frame | 18 | `src/nek_post/vortex_3d_workflow.py:41–49,119–121` |
| vortex_series | 20 | `src/nek_post/vortex_3d_series.py:84–89` plus vortex_frame |
| lambda_ci_slices | 5 | `src/nek_post/lambda_ci_series.py:123–126` |
| omega_x_native | 6 | `src/nek_post/omega_x_workflow.py:109–113` |
| omega_x_p7 | 6 | `src/nek_post/omega_x_p7_series.py:81–85` |

來源：[source_fingerprints.json](../source_fingerprints.json)。下列 basename 對應現有 `src/nek_post/` 位置，整個檔案 bytes（含 whitespace/comments/imports）都保護，不能換 wrapper 或重排 imports。

| Protected source | SHA-256 | Groups |
| --- | --- | --- |
| `fields.py` | `f2a668f48ef3b4a28363fa5e9daf496653235783d81ca0c7564855f723ca3346` | vortex_frame, vortex_series |
| `gll.py` | `b5d0a191ce11f6aabdc36ebc8c4e0db8313c6404509e5f1cd489f7e2997a032b` | vortex_frame, vortex_series, lambda_ci_slices, omega_x_native, omega_x_p7 |
| `gll_directional_integration.py` | `f946f049147dd0d686d84c99b38296923743a6de4e8a45cef8f743c9e73c959f` | vortex_frame, vortex_series, omega_x_p7 |
| `gll_refinement.py` | `fb9d26ae6ca95c85dc659f283e11af340f79182b9d4e29b4c8c1d1adfac0ef8a` | vortex_frame, vortex_series |
| `hex_validation.py` | `dd5ec26bedc90440253927d6f8b870d3676ad7b13207b94c3c343811340e7711` | vortex_frame, vortex_series |
| `io_nek.py` | `0e194625cc2ad85d2aaf0a8b5d38a9433889cf09e94aa9fd8d9f7040541bb7da` | vortex_frame, vortex_series |
| `l2_projection.py` | `1254d7ea6b54ada2540ce438e5b589970f079a92de4572a1ed02c80ebc795934` | vortex_frame, vortex_series, lambda_ci_slices, omega_x_native |
| `lambda_ci_plotting.py` | `eca2ad6b713044dbc4e9e1c4b8fe9115c722c18328a6d9e2f549bf183176997e` | lambda_ci_slices, omega_x_p7 |
| `lambda_ci_series.py` | `cf0dd4449ded02ef48466831843085c2e4d0dc4eb91e1cd726f5e6cde0ca2bb4` | vortex_frame, vortex_series, omega_x_p7 |
| `lambda_ci_slice_export.py` | `b7d5d9cedad7c475df9913778725b2dff4f59da02431bba1c75b038c713d5b45` | vortex_series |
| `lambda_ci_workflow.py` | `4727e9a404f8465a37e789a9bd4049314b48a28a17952fc08bce3f23da5549ad` | vortex_frame, vortex_series |
| `omega_x_p7_series.py` | `17228defc3aa972c9ee5934c7319ce15305eacd30f3f8bfebd5a143086e14d6a` | omega_x_p7 |
| `omega_x_p7_slice.py` | `13c89cd77ab78f95656dc811110685837d63d1ee9da71c0a98a3aba9cab3e736` | omega_x_p7 |
| `omega_x_workflow.py` | `8b7b441f7eb45d7deeb3159b98d256502231b0377151ae1fa7214997241d6176` | omega_x_native |
| `rectilinear_export.py` | `bb75dfce824705bcbf3724048ae369879f781a61f0abc9d11c1bed6cf781088e` | vortex_frame, vortex_series, omega_x_native |
| `spectral_derivatives.py` | `326bf2c73b8dda1826c6b3598c1f004327fc923512c1ccc7ffa6f86107387981` | vortex_frame, vortex_series, lambda_ci_slices, omega_x_native |
| `spectral_interpolation.py` | `40b597e252ac3bbc366e05306fb37c473a06ad4a7b35f00a0472e6902a04db04` | vortex_frame, vortex_series |
| `swirling_strength.py` | `722d35c6427cc76297351473ed1be9661771c1941de06df7aaefa01a877adfe5` | vortex_frame, vortex_series, lambda_ci_slices |
| `vortex_3d.py` | `897fddca7a2568bb54fd8642ec1891b774e14a83f384257f50bc46dfa5ea8dff` | vortex_frame, vortex_series |
| `vortex_3d_artifacts.py` | `6c6bd4f4efe2828711e446a876fbb1d77d8e82bcce2d44ba31dc11eca933d68b` | vortex_frame, vortex_series |
| `vortex_3d_reconstruction.py` | `257246a28d5056c763cd5b69827dbb187458b787e592b25eb8bdec5f7861ad3c` | vortex_frame, vortex_series |
| `vortex_3d_selection.py` | `1184d7eb2be28ea3182d073e52c138d93760b161d19d6f6361658843f76538c3` | vortex_frame, vortex_series |
| `vortex_3d_series.py` | `058013a218944e263651c96b49712391fd45d197e9d31bc044247f47f5bece2b` | vortex_series |
| `vortex_3d_workflow.py` | `3f04e61361af488dd23ae1e6445360a7f07d9f30e792fb773ac9591f1ea0f8a3` | vortex_frame, vortex_series |
| `vorticity.py` | `978ec2f80f32cce3b8edb4db41e438a05c276c18f38f404a6a181bd50dae5578` | omega_x_native |

額外 `src/nek_post/vtu_compat.py` identity：`7d0f29d16d1b3919da5249e53120e4d8ddec0eec295a5cda85b8372e099d8080`，與 accepted HEAD bytes 相等；own implementation check at `:504,544–545`。它不在五組 union 內，不因此可改。

Module-qualified workers/plans 的位置、名稱、globals 與 serialized/spawn identity 保留，例如 `leading_edge_parallel.py:35,50,225,248`、`spectral_horizontal_slice.py:37`、`refined_gll_horizontal_slice.py:45`。新 unsigned helper 不得拿來規避 fingerprint 或改 scientific behavior。

Native strict-resume 另驗 source/input/output/configuration/software/Python/artifact/control identities（`vortex_3d_series.py:177–204,342–371,574–606`）。任何失配按 native policy 拒絕，不改 metadata/signatures、移動資料或放寬 validator 偽造相容。

## Mandatory future gates

| Gate | Required evidence / failure boundary |
| --- | --- |
| G-FREEZE | Before/after five exact member/hash maps, 25 complete sources and sibling layout, vtu own identity, worker/plan module names；失配停止，不修 capture |
| G-HELP | All root/group/49 leaf helps in minimal environments; no handler imports/config reads/application allocation; unavailable leaves fail clearly on execute |
| G-CONFIG | Per-file precedence and invalid override errors; invocation/canonical-YAML symlink semantics; logical resources in wheel/sdist; unchanged typed scientific selection; separate input/output views plus full original safety inventory in child paths YAML; every writing leaf needs explicit/env machine paths with no fallback; effective YAML reread/safety-identity/hash change rejection |
| G-PLAN | Root absent and existing-root snapshots show zero allocation/change: no YAML materialization, logs, temp, Matplotlib caches, locks or children; plan gives exact argv/destinations and Unknown refusal |
| G-WRITE | Every writing leaf first passes mandatory Filesystem Safety Context before allocation/write-capable feature imports; test the safety refusal cases below. Every enabled leaf/flag branch: observe complete final/temporary/companion/cache/backup/lock/delete/write set against allowlist; closure of every **Unknown** row; parent creation, inherited cache env, external tools and failures included; only two explicit P exceptions; V read-only in publish |
| G-EXEC | Installed children from unrelated cwd with no checkout/PYTHONPATH help; correct interpreter/argv/config/options/cwd/env; stdout/stderr split; first failure and signal mapping; process-group cleanup/reaping; no scientific children in installed dry-run; C11/C12 benchmark child keeps parent R and exact contained D override, no extra nesting, warmup/measured CSV readback and unchanged schemas/equality/default behavior |
| G-LEGACY | All 46 unchanged paths/implementations; exact flags/defaults/choices/errors/globals/aliases/monkeypatch behavior; script 03 three branches and 07 dedup/order characterized before adapting |
| G-NUMERIC | Independent analytic/reference expectations; preserve algorithm/operation ordering, sample geometry/masks/weights, dtype/order/time and serialized schema; compare arrays/metadata, not only hashes/filenames/test counts |
| G-LIFECYCLE | Per-workflow collision/skip/overwrite/native locks, under-lock revalidation, failed/incomplete stage retention, eligible/ineligible resume/recovery, source/software/path/config change rejection, PVD actual times and no-replace publication; P7 series-wide persistent lock plus private PVD temporary/hard-link, no extra per-PVD lock |
| G-RESULTS | Read-only inventory; fresh catalog/preview; inventory_only remains provisional; P/V authorization; reviewed-versus-fresh catalog equality; exclusive two-file publish and preserved partial failure; scientific artifacts unchanged |
| G-PLATFORM | Actual feature/dependency/tool versions; mandatory Poppler/GNU time/PyVista; Linux renameat2/fcntl/resource/hard-link/fsync behavior on intended filesystem; supported platform boundaries explicit |
| G-REVIEW | Complete diff including untracked files, relevant tests, unchanged artifacts/source identities and unresolved issues independently reviewed per batch/checkpoint; no self-acceptance |

Mandatory B2/G-WRITE safety cases（**FUTURE TEST / NOT RUN**）：對所有 writing routes，包括 42、43 catalog build、44 preview/publish 與 Extra，測 missing `--paths-config`/env source、invalid/empty/incomplete case inventory、selected/unselected raw-case R、historical P/H overlap、symlink/unsafe ancestors、explicit input/output overlap。每個 refusal 均須證明沒有 R/.cli/log/cache/lock/staging/temp allocation、沒有 write-capable handler imports；並測合法 R、僅兩個 P targets 的例外、V 唯讀，以及 help/plan/三個 read-only leaves 零寫入。保留原始/derived H safety inventory，output translation 不得隱藏歷史 roots；context 不完整時 route unavailable。

Mandatory C11/C12 handshake cases（**FUTURE TEST / NOT RUN**）：以 [exact argv contract](target_architecture.md#exact-benchmark-child-output-handshake-m1) 驗 checkout-free installed children、完整 argv/config/scientific values、D 直接放置 CSV pair/no extra nesting、R 外 override refusal、native parent-created D 與 logs collision boundaries。Warmup cleanup 前及 measured repetitions 都讀回 expected filenames/schema 並驗 artifact equality；不宣稱現有 warmup 已做該 readback。Legacy None-default builder/cwd/env/monkeypatch、native cleanup/overwrite 與 protected-source freeze 保持相同。

G-NUMERIC 必須分別覆蓋 griddata 與 SEM、sampled L2 與 GLL recovery、external 與 concentration front、Cantero 與其他 front definitions、rightmost 與 Moore、native 與 tensor-first lambda、native 與 saved-scalar P7 omega、native/reconstructed VTU 與另案 VTI。不以其中一組通過推論另一組。

## Documentation-only quality checks

| Check | Required structural assertion |
| --- | --- |
| Eight ADRs | 0001–0008 files all exist; Context, Alternatives, Decision, Rationale, Trade-offs, Compatibility consequences, Validation requirements, Deferred decisions each substantive |
| Legacy coverage | All actual 46 scripts appear in migration matrix and exactly one C batch; no unsupported delegate symbols |
| CLI coverage | Exactly 49 unique leaves, three intentional splits and integrate gll-field |
| Destination coverage | Same 49 keys; writing routes have destination/ancillary/native-policy entries or explicit Unknown plus G-WRITE; read-only routes allocate no R |
| Authorization | Every writing leaf FS context from explicit/env machine paths before allocation/imports, independent of scientific config; R primary/operational roots; only navigation publish two-file P exception; V read-only input in publish and authorized writing preview under R; benchmark exact contained D override without extra nesting |
| Roadmap | B1–B3, C1–C22 and D1–D3; C7 and C12 dependencies; C16/C18/C19 lifecycle checkpoints; per-batch review/rollback/branch proposals |
| Claims | New CLI/hooks/resources/tests remain proposed/not implemented; no statement that unexecuted runtime/wheel/scientific tests passed |
| Scope/integrity | Only new Markdown below phase4; all original tracked file SHA-256 values and existing branch refs unchanged; raw/results bounded metadata equal; protected groups and vtu unchanged |
| Format | Local Markdown targets exist; tables/fences consistent; git diff --check and unstaged new Markdown whitespace checks |

External metadata comparison covers directory entries, lstat modes/sizes/mtime_ns and symlink targets for the seven case roots and results. It is **not** full payload hashing, historical authenticity proof or a scientific validation run. Filename count verifies N11=101/others=81 only; actual physical-time alignment remains a runtime gate.

## Unknowns / deferred decisions

Full transitive write-set evidence is explicitly Unknown for the marked [destination rows](destination_contracts.md), including process/tool caches. All such routes remain unavailable until implementation closes G-WRITE. External callers/pickles, actual wheel/platform success, environment-dependent historical strict-resume eligibility and complete runtime behavior are unverified.

Exact handler filenames/private interfaces, signal escalation timings, resolved config semantic-hash encoding and sensitive-environment record policy are B-stage implementation decisions under these contracts, not authorization to relax them. Final results_v2 schema belongs to Phase 4E. Retained branch numerical/VTI capability decisions remain separate.
