# 網格細化結果：如何選擇分析入口

本指南使用結果根目錄 `/data/Nek5000_data/results`。連結指向現有產物，
不會啟動計算。[總覽](README.md)、[生產程式證據](producer_scripts.md) 與
[歷史分類](historical_results.md) 提供其他入口。科學定義見
[Phase 1 方法](../methods/vortex_numerical_method.md)。

**VERIFIED FROM EXPLICIT METADATA**：下表以實際控制檔、PVD 及 CSV 為證據。
所有類別都是後處理結果；「native」表示原生節點上的衍生場匯出。
原始速度／濃度快照位於結果樹之外，來源依 sidecar 的 `source_file` 查核。
N7_H、N7_VH、N7_VVH 的 P7 候選共同階數亦由
[研究設定](../../config/h_refinement.yaml) 支持；名稱本身不證明網格或物理條件。

| 類別、變數與案例 | 主入口、格式與證據 | 使用方式與限制 |
| --- | --- | --- |
| field_comparison：三案例的濃度、速度大小、壓力擾動 | [run.json](/data/Nek5000_data/results/h_refinement/field_comparison/run.json)、[inventory.json](/data/Nek5000_data/results/h_refinement/field_comparison/inventory.json)、[field_errors.csv](/data/Nek5000_data/results/h_refinement/field_comparison/field_errors.csv)；snapshot 子目錄保存 NPZ／JSON | run 記錄 y=0.75、500 × 200 共用 x-z 格點、目標時間 5/10/15/19.5，以 N7_VVH 為數值參照；未作時間插值。先讀 selected_times.csv，不能用相同幀編號當作相同物理時間。 |
| convergence_analysis：H/VH/VVH 差異、方向網格間距及形態診斷 | [analysis.json](/data/Nek5000_data/results/h_refinement/convergence_analysis/analysis.json)、[方向解析度 CSV](/data/Nek5000_data/results/h_refinement/convergence_analysis/directional_mesh_resolution.csv)、[分類 CSV](/data/Nek5000_data/results/h_refinement/convergence_analysis/convergence_classification.csv)、[差異圖](/data/Nek5000_data/results/h_refinement/convergence_analysis/pairwise_differences.png)；CSV／PNG | analysis 的 source_sha256 連結既有比較產物，source_artifacts_unchanged 回報來源未變。方向間距是實際幾何證據；不可由 H/VH/VVH 推定固定細化比。分類表記錄時間誤差、非單調及缺乏單一 scalar h 等限制，observed_order 空值不得補猜。N7_VVH 是可用最細數值參照，並非精確解。 |
| cantero_mean_front：三案例等效高度門檻前緣 | [N7_H 時間序列](/data/Nek5000_data/results/h_refinement/cantero_mean_front/N7_H/N7_H_cantero_mean_front_timeseries.csv)；N7_VH/N7_VVH 位於同層案例目錄 | CSV 提供抽取結果。方法由 scripts 23/24 的 composite GLL z/y 積分及高度門檻定義支持；這個前緣不是固定 z 平面的 leading edge。個別歷史執行版本仍 UNKNOWN。 |
| cantero_re3450：三案例重建前緣與 Cantero 比較 | [摘要 CSV](/data/Nek5000_data/results/h_refinement/cantero_re3450/cantero_re3450_multicase_summary.csv)、[線性疊圖](/data/Nek5000_data/results/h_refinement/cantero_re3450/cantero_re3450_multicase_linear_overlay.png)、[相對 N7_VVH 表](/data/Nek5000_data/results/h_refinement/cantero_re3450/cantero_re3450_relative_to_N7_VVH_timeseries.csv) | 摘要明列 front_source、paper_source、moving_average 與有效 window=11；圖表使用重建後軌跡。參照對齊只在物理時間重疊內進行；原始與重建前緣不可混稱。詳見[既有方法](../h_refinement_cantero.md)。 |
| leading_edge：三案例固定平面前緣與抽取方法差異 | [Step 6B summary](/data/Nek5000_data/results/h_refinement/leading_edge/step6b/comparison/summary.json)、[refined-GLL 報告](/data/Nek5000_data/results/h_refinement/leading_edge/refined_gll_nodes10/figures/h_refinement_refined_gll_nodes10_report.json)；raw CSV／NPZ、comparison CSV／PNG／PDF | Step 6B 報告定義共同時間、mask、失敗 crossing 與來源雜湊。另一套 case-native GLL10 圖把原始 8 節點 P7 解取樣至 10 節點；各案例 y 網格不同，報告明示未作跨案例逐點 RMS。Moore-boundary 與 rightmost-crossing 必須分別查找。step6a_smoke 是測試範圍。 |
| omega_x：N7_H native GLL 流向渦度 | [series PVD](/data/Nek5000_data/results/h_refinement/omega_x/case_N7_H/case_N7_H_omega_x_series.pvd)、[frame-79 JSON](/data/Nek5000_data/results/h_refinement/omega_x/case_N7_H/case_N7_H/f00079/case_N7_H_f00079_omega_x.json)；VTR／PVD／JSON | omega_x = ∂w/∂y − ∂v/∂z，作 scalar mass-weighted recovery。JSON 的 native_gll=true、interpolation=none 指原生節點匯出；線性 VTK 單元不等於增加 SEM 自由度。保留既有重複 case_N7_H 路徑。 |
| omega_x_highorder_slices：N7_H 已儲存渦度的 P7 重建 | [series PVD](/data/Nek5000_data/results/h_refinement/omega_x_highorder_slices/case_N7_H/case_N7_H_omega_x_p7_slice_series.pvd)、[frame-79 validation](/data/Nek5000_data/results/h_refinement/omega_x_highorder_slices/case_N7_H/f00079/case_N7_H_f00079_omega_x_y0p75_xgt0_n29_validation.json)；VTR／PNG／PDF | y=0.75、x>0、每元素邊 29 點；diagnostics 記錄 polynomial_degree=7、native_nodes_per_element=8。這是已投影 omega_x 標量的重建，與 lambda_ci 的 tensor-first 重建路徑不同。歷史 frame-79 缺少插值程式 signature 的限制見[系列文件](../omega_x_p7_slice_series.md)。 |

## lambda_ci：先選 native、ParaView 或 Python-only

**VERIFIED FROM EXPLICIT METADATA**：
[native frame-79 JSON](/data/Nek5000_data/results/h_refinement/lambda_ci/case_N7_H/f00079/case_N7_H_f00079_lambda_ci.json)
把原生 8 × 8 × 8 節點、投影與 lambda_ci 統計連到
[native VTR](/data/Nek5000_data/results/h_refinement/lambda_ci/case_N7_H/f00079/case_N7_H_f00079_lambda_ci.vtr)。
此證據只建立這一幀 native 體積入口，不能推出完整 native 體積系列。

| 產品 | 實際 metadata 與入口 | 取樣證據 |
| --- | --- | --- |
| ParaView 重建切片 | [frame-79 ParaView JSON](/data/Nek5000_data/results/h_refinement/lambda_ci/case_N7_H/f00079/plots/case_N7_H_f00079_lambda_ci_high_order_slices_paraview.json)、[y VTR](/data/Nek5000_data/results/h_refinement/lambda_ci/case_N7_H/f00079/slices/case_N7_H_f00079_lambda_ci_y_midplane.vtr)、[series summary](/data/Nek5000_data/results/h_refinement/lambda_ci/case_N7_H/processing_summary.json)、[y-series PVD](/data/Nek5000_data/results/h_refinement/lambda_ci/case_N7_H/case_N7_H_lambda_ci_y_midplane_series.pvd) | output_paths 明確連到 VTR；29 點 = 28 subdivisions + 1。y 切片尺寸 3809 × 1 × 225；x=15 切片為 1 × 337 × 225。PVD、summary 與 frame-79 metadata 相符。 |
| Python-only 重建切片 | [32-point JSON](/data/Nek5000_data/results/h_refinement/lambda_ci/case_N7_H/f00079/plots/case_N7_H_f00079_lambda_ci_high_order_slices.json)、[y 圖](/data/Nek5000_data/results/h_refinement/lambda_ci/case_N7_H/f00079/plots/case_N7_H_f00079_lambda_ci_y_midplane_xge0p0.png) | samples_per_element=32；輸出映射為獨立 PNG／PDF／NPZ，沒有上述 VTR。y samples shape 為 249 × 4217。 |

**VERIFIED FROM SOURCE CODE**：依 [Phase 1 方法](../methods/vortex_numerical_method.md)，
native 路徑為 local velocity gradients → project 九個 tensor components → evaluate lambda_ci；
重建路徑在 eigensolve 前重建九個已投影分量。
直接對 native lambda_ci 標量插值一般不等價。omega_x 與 lambda_ci 可協助紊流渦結構研究，
但目錄盤點不驗證科學結論。

## vortex_3d_series：N7_H 三維時間序列

**VERIFIED FROM EXPLICIT METADATA**：
[PVD](/data/Nek5000_data/results/h_refinement/vortex_3d_series/gll10/case_N7_H/series/case_N7_H_vortex_3d_series.pvd)
包含 81 幀，物理時間 0–20；
[manifest](/data/Nek5000_data/results/h_refinement/vortex_3d_series/gll10/case_N7_H/series/manifest.json)
記錄 frame index、time 及來源／sidecar／VTU 雜湊。
[frame 79 binary VTU](/data/Nek5000_data/results/h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.vtu)
的 [sidecar](/data/Nek5000_data/results/h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.json)
記錄 time=19.50240408807、來源 P7（8 節點）、GLL10 取樣、omega_x/lambda_ci
及 x=0–17、y=0–1.5、z=0–1 範圍。GLL10 沒有提高原模擬解析度。
summary 回報完成不代表與 production Nek5000 執行檔精確等價。

**VERIFIED FROM SOURCE CODE**：[strict resume](../../src/nek_post/vortex_3d_series.py)
檢查配置、signature、frame 身分與控制紀錄。勿手動更名、移動、插入檔案或改寫 manifest。
本階段不重新跑 resume／匯出，不解碼大型 VTU。

深入查找時，在已發布 results_index.csv 篩選 study=h_refinement、analysis 及 case_id；
variant、metadata_sources、notes 可補足 case_id 空白的跨案例報告。
所有記錄仍為 inventory_only；未提供 producer/revision 的歷史執行保持 UNKNOWN。
