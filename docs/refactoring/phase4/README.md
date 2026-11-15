# Phase 4A：Architecture documentation

## 狀態、範圍與基準

**DOCUMENTATION IMPLEMENTED / INDEPENDENT DOCUMENTATION REVIEW PENDING**。
Conservative Extension 是已接受的架構方向；M1–M5 在 targeted re-review 中於 design level 解決，沒有建立新的 blocking contradiction。該 review 是 PASS WITH CONDITIONS：完整 ADR 與 route/batch schedules 要保存為 repository docs 並獨立審閱。本次文件補上這些內容，**不宣稱條件已獨立驗收**。

固定 source baseline：`3f9be2e4cd088c439700ba857e344adde9025712`（dev）。文件 branch：`feature/phase4a-architecture-docs`；master 保護。本文件集沒有建立 CLI、改 Python/config/tests、安裝 wheel、執行 pytest 或科學工作流程。

## 文件導覽

| Document | 唯一責任 |
| --- | --- |
| [Target architecture](target_architecture.md) | Package/dispatch/installed execution、scientific boundaries |
| [CLI migration matrix](cli_migration_matrix.md) | 46 exact legacy paths、49 proposed leaves、existing delegates、risk/tests/batch |
| [Destination contracts](destination_contracts.md) | 每個 leaf 的 R/root translation、write effects、P/V exception、Unknown write sets |
| [Configuration and dependencies](configuration_dependencies.md) | 配置 precedence/path semantics/resources、per-leaf prerequisite profiles/help/backend |
| [Implementation roadmap](implementation_roadmap.md) | B1–B3、C1–C22 exact script assignment/prerequisites/review/rollback、D1–D3 |
| [Acceptance gates](acceptance_gates.md) | Evidence labels、protected-file lists、future tests、documentation structural checks |
| [ADR-0001](adr/0001-package-topology.md) | Conservative Extension package topology |
| [ADR-0002](adr/0002-cli-dispatch.md) | Static argparse dispatch and installed children |
| [ADR-0003](adr/0003-legacy-compatibility.md) | Legacy paths/implementations and characterization |
| [ADR-0004](adr/0004-numerical-identity.md) | Complete source/runtime identity |
| [ADR-0005](adr/0005-configuration-dependencies.md) | Portable configuration/resources/dependencies/backend |
| [ADR-0006](adr/0006-filesystem-policy.md) | Per-route destinations, native lifecycle and publication exception |
| [ADR-0007](adr/0007-results-provenance.md) | Historical results, execution provenance, deferred results_v2 |
| [ADR-0008](adr/0008-retained-branches.md) | Separate retained scientific capabilities |

## Evidence vocabulary

- **VERIFIED BY SOURCE / AST**：在固定 baseline 靜態查核；不是 runtime proof。
- **PROPOSED / NOT IMPLEMENTED**：未來 CLI/components/contracts，不能當作可用命令或現有 API。
- **HISTORICAL**：使用者提供的 2,800 passed / 136.89 s 與 94 inventory-only bundles；本次未重跑或重新 catalog。
- **UNKNOWN**：未證明的 full transitive write sets、外部 callers、environment/platform eligibility；以 mandatory acceptance gate 封鎖其啟用。
- **FUTURE TEST / NOT RUN**：wheel/editable、help/runtime、numerical、serialization、failure/recovery、publication 等測試均未於此次執行。

## Research 邊界

P-study：N5/N7/N9/N11，以 configured N11 為 P comparison reference（[cases.yaml](../../../config/cases.yaml) lines 1–7,18–54）；不是 exact solution。H-study：N7_H/N7_VH/N7_VVH，N7_VVH 是 provisional finest available reference，必須通過 inventory/order/domain/mesh/time checks（[h_refinement.yaml](../../../config/h_refinement.yaml) lines 1–7）；element count 不單獨證明 local refinement 或 asymptotic convergence。

本次只用 filenames 做 bounded snapshot inventory：N11 101，其餘六案例各 81 個 GC0.fNNNNN；未讀 headers/solution payload 驗證全部實際時間。Cross-case comparison 要用 case-specific indices 與 actual physical times，不用相同 frame number 對齊。GC8950_N7 是特定 Re8950 front capability 的額外 configured input，不混入上述七案例 snapshot-count statement。

Energy 保持 existing P-refinement scope。Scattered griddata/SEM、sampled L2/GLL recovery、external/concentration front、Cantero/其他 front、rightmost/Moore、native/tensor-first lambda、native/saved-scalar P7 omega、VTU/uniform-volume VTI 都保持不同定義。Final results_v2 留到 Phase 4E。

## 授權與下一步

下一步是 complete documentation diff 的獨立審查。文件內容不授權 branch/commit/merge/push、刪除 retained branches、修改 scientific artifacts 或開始 Phase 4B。所有 46 legacy implementations 在 Phase 4 保留不變；不是 wholesale relocation。文件簽核與各後續 implementation task 另行控制。
