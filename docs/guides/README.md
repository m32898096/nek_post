# 依任務尋找既有工作流程

此導覽建立於 Phase 3B Batch 1，最初整理能源收支入口、模組責任及相容性測試。
目前依研究任務連結既有指南；執行前請先讀各指南的輸入、輸出保護與限制。

| 任務 | 既有說明 |
| --- | --- |
| 開始設定與比較工作流程 | [Workflow](../workflow.md)、[完整腳本索引](../script_index.md) |
| 能源收支診斷 | [script 12 使用方式、輸入/輸出與相容性](../energy_budget_closure.md) |
| 前緣與時間演化 | [Front kinematics](../front_kinematics.md)、[Leading-edge evolution](../leading_edge_evolution.md) |
| 網格細化研究 | [H-refinement study](../h_refinement_study.md) |
| 渦流方法與來源限制 | [Numerical methods](../methods/README.md) |
| 獨立數值參照驗證 | [Phase 1B results](../methods/phase1b_results.md) |
| 3D 渦流輸出與 strict resume | [Binary VTU guide](../vortex_3d_binary_vtu.md) |
| 尋找與解讀已發表結果 | [研究結果導覽](../results_catalog/navigation_zh_tw.md)、[catalog 說明](../results_catalog/README.md) |
| 架構、相容性與重構紀錄 | [稽核與實作紀錄索引](../refactoring/README.md) |

執行命令前依各指南確認輸入、輸出與覆寫條件。導覽連結與通過測試不代表已重跑
production 工作流程，也不代表與 production Nek5000 executable 完全對等。
後續變更可參照[已審查的重構計畫](../refactoring/refactor_plan.md)與實作紀錄；
計畫中的提案需依各批授權及實際完成範圍解讀。

[Batch 2 腳本相容性契約](script_compatibility_contracts.md)說明 script 08/13 的
CLI、暫存輸出、獨立參考與 protected-source guard，以及尚未驗證的介面。
