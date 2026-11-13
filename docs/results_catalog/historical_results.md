# 歷史、診斷與不確定結果

這是文件層級的用途分類，供研究者定位證據，不會改寫 CSV 的 artifact_class、status
或 validation_level。舊不等於無效，診斷不等於錯誤，缺 provenance 不等於不可使用，
reported_completed 不等於科學驗證。本指南不提出刪除或搬移清單。

| 分類 | 現有例子與支持證據 | 可得結論與不能推出的結論 |
| --- | --- | --- |
| CURRENT / PRIMARY（文件化工作流程入口） | [vortex manifest](/data/Nek5000_data/results/h_refinement/vortex_3d_series/gll10/case_N7_H/series/manifest.json) 與 [系列文件](../vortex_3d_binary_vtu.md)；[lambda_ci series summary](/data/Nek5000_data/results/h_refinement/lambda_ci/case_N7_H/processing_summary.json) 與 [切片文件](../lambda_ci_highorder_slices.md) | VERIFIED FROM EXPLICIT METADATA：控制檔可識別案例、配置、幀及來源；是閱讀入口，不表示其他取樣產品過時，也不建立 matched production solver equivalence。 |
| HISTORICAL（有明確歷史說明） | [omega_x P7 系列文件](../omega_x_p7_slice_series.md) 明確描述 frame-79 歷史 JSON 缺少 interpolation-code signature，並說明重用時的獨立相容性檢查；[frame-79 sidecar](/data/Nek5000_data/results/h_refinement/omega_x_highorder_slices/case_N7_H/f00079/case_N7_H_f00079_omega_x_y0p75_xgt0_n29_validation.json) | VERIFIED FROM SOURCE CODE／既有方法文件支持歷史格式區別；目前不重新執行相容性檢查，亦不補寫 signature。 |
| HISTORICAL 候選，確切年代 UNKNOWN | [archive/moore_boundary](/data/Nek5000_data/results/poly_order_compare/archive/moore_boundary/)、[same_index_f00080](/data/Nek5000_data/results/poly_order_compare/archive/same_index_f00080/) | INFERRED FROM CONVENTIONS：archive 路徑與其中 CSV、圖、log 是保留產物的線索；缺完整執行歷史，不能確定已被哪個工作流程取代。相同 index 不保證相同物理時間。 |
| DIAGNOSTIC / TEST | [refined-GLL smoke report](/data/Nek5000_data/results/poly_order_compare/leading_edge_gll_refinement/smoke_report.json)、[validation report](/data/Nek5000_data/results/poly_order_compare/leading_edge_gll_refinement/validation_report.md)、[worker benchmark summary](/data/Nek5000_data/results/poly_order_compare/leading_edge_benchmark/N7/leading_edge_workers_summary.csv) | VERIFIED FROM EXPLICIT METADATA：報告列 smoke frames、取樣及比較，benchmark 表記錄效能用途；測試範圍不自動代表完整 production 驗證。 |
| DIAGNOSTIC / TEST 候選 | [step6a_smoke](/data/Nek5000_data/results/h_refinement/leading_edge/step6a_smoke/)、[N7_layout_smoke](/data/Nek5000_data/results/poly_order_compare/front_detection/N7_layout_smoke/)、[energy smoke](/data/Nek5000_data/results/poly_order_compare/energy_budget_closure_smoke/N7/)、[fix test](/data/Nek5000_data/results/poly_order_compare/leading_edge_fix_test/N7_moore_interpolated_full/) | INFERRED FROM CONVENTIONS：名稱支持用途線索，仍需檢視個別控制／log；不宣稱每個檔案已完成驗證。 |
| INCOMPLETE / AMBIGUOUS（適用條件，非強制套用） | 盤點遇到缺失 declared output、unsupported output schema、破損 PVD 或衝突狀態時，保留具體 notes。現有 [CSV](/data/Nek5000_data/results/results_index.csv) 沒有此 status 的列 | 不為了填滿分類而新增實例；JSON inspection cap 外的檔案仍可能有未檢出的問題。 |
| UNKNOWN（來源／關聯不充分） | [spectral_volume PVD](/data/Nek5000_data/results/poly_order_compare/spectral_volume/N7/N7_GC0.pvd) 的取樣設定；[energy summary](/data/Nek5000_data/results/poly_order_compare/energy_budget_closure/energy_budget_closure_summary.csv) 與同目錄三份案例表的共同執行歷史 | 表與 PVD 可讀，不足以補出 producer/revision。energy summary 只含 N7，不據此判定 N5/N9 表失效。 |

## 目前盤點觀察與研究者決策

2026-11-13 的 94 個 provisional bundles 中，CSV 的 artifact_class 為
70 unclassified、12 smoke_test、7 diagnostic、5 historical；status 為
92 unknown、2 reported_completed，全部 validation_level=inventory_only。
這是分組規則的輸出：archive 裡的 smoke/fix_test 可優先歸為 smoke_test，
因此不能把這些數字當作互斥的科學有效性統計。

**UNKNOWN**：全部 94 列的 producer_script 與 code_revision 都沒有可用明確欄位值。
數值模組 SHA、source_sha256、output_sha256 與 raw_hashes_unchanged
可建立局部來源關係或內部一致性，不等於完整的歷史命令或經認證的 Git revision。
色盤 JSON 陣列、超過 1 MiB 的 relocation 控制與每群組 64 JSON 上限皆有盤點限制。

29 點 ParaView lambda_ci 與 32 點 Python-only 切片是可區分的產品，不能只按年代
把後者歸為 obsolete。實際映射見 [場產品指南](h_refinement.md)。
rightmost、Moore、Cantero equivalent-height 與 front_simple.dat 所述前緣也不是同一觀測量。

選用結果前，先找研究問題所需的變數與物理時間，再查 output mapping、原方法與 notes。
找不到證據時記為 UNKNOWN；需重新研究來源時，在結果樹之外保留新證據。
不手動改寫 scientific JSON、manifest、LOCK 或 checkpoint 以消除不確定性。
