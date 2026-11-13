# 分析與程式：能力證據與歷史執行證據

**VERIFIED FROM SOURCE CODE**：下表連結目前 CLI 與實作，表示可支援的工作流程。
它們皆屬 **CAPABLE / LIKELY SCRIPT**，不是既有檔案的 **CONFIRMED PRODUCER**。
目前 CSV 的 producer_script/code_revision 全為空；本階段沒有把任何 entrypoint
認定為已確認歷史 producer。manifest 的 workflow、output_paths、source hashes
可確認資料關係，通常仍不足以證實哪一次 CLI invocation。

| 分析／輸入 → 輸出 | 能力入口與相關模組 | 使用方式與證據界線 |
| --- | --- | --- |
| 原始快照 → slice／共用格點誤差與圖 | [02](../../scripts/02_extract_midspan_slice.py)、[03](../../scripts/03_compare_poly_orders.py)、[04](../../scripts/04_plot_summary.py)；[comparison](../../src/nek_post/comparison.py) | NPZ、CSV、PNG；由 comparison-set 設定選 index。對應 figures/tables 的命名能力已由程式確認，歷史執行 UNKNOWN。 |
| energy_budget.dat → closure CSV／PNG | [12](../../scripts/12_check_energy_budget_closure.py)；[energy_budget](../../src/nek_post/energy_budget.py) | `--cases N5,N7,N9` 為目前語法；不能當作現有 N7-only summary 的歷史命令。 |
| front_simple.dat → 速度、重建位置與 paper overlays | [13](../../scripts/13_front_kinematics.py)、[14](../../scripts/14_fig5a_paper_overlay.py)、[15](../../scripts/15_processed_xt_paper_overlay.py)；[front_io](../../src/nek_post/front_io.py) | CSV／PNG；外部前緣資料的原始生產定義 UNKNOWN。 |
| 濃度快照 → detected front 與 paper 比較 | [16](../../scripts/16_detect_front_from_concentration.py)、[17](../../scripts/17_fig5a_detected_front_overlay.py)、[18](../../scripts/18_n7_reconstructed_xt_fourway_overlay.py)；[front_detection](../../src/nek_post/front_detection.py) | CSV／PNG／診斷；linear/spectral 及 Re3450/Re8950 輸入不可混接，逐一核對 CLI 的 input 選項。 |
| 快照 → spanwise leading edge／圖／worker benchmark | [19](../../scripts/19_compute_leading_edge_evolution.py)、[20](../../scripts/20_plot_leading_edge_evolution.py)、[21](../../scripts/21_benchmark_leading_edge_workers.py)；[leading_edge_artifacts](../../src/nek_post/leading_edge_artifacts.py) | CSV／JSON／PNG／PDF；19 抽取、20 讀保存表格、21 測試 workers。現有 singular benchmark 目錄不證明由目前 plural 預設直接產生。 |
| 濃度 → equivalent height → mean front → reconstructed paper comparison | [23](../../scripts/23_compute_cantero_equivalent_height.py)、[24](../../scripts/24_compute_cantero_mean_front.py)、[25](../../scripts/25_reconstruct_cantero_front.py)、[26](../../scripts/26_overlay_cantero_re3450_multicase.py)；[cantero_mean_front](../../src/nek_post/cantero_mean_front.py) | 24 支援 H 案例與明確 output root；25 正式案例限 N7；26 用 `--cases N7_H N7_VH N7_VVH` 可選 h 組。輸入／paper 來源可由 summary 證實，entrypoint 的歷史執行 UNKNOWN。 |
| h 快照／保存共用格點 → 場比較與收斂診斷 | [27](../../scripts/27_inventory_h_refinement.py)、[28](../../scripts/28_extract_h_refinement_slice.py)、[29](../../scripts/29_compare_h_refinement_fields.py)、[30](../../scripts/30_analyze_h_refinement_convergence.py)；[h_refinement_analysis](../../src/nek_post/h_refinement_analysis.py) | JSON／CSV／NPZ／PNG。30 對保存 Step 3 產物作分析並記 source hashes；不能把條件式 observed order 當作既定收斂率。 |
| h leading edge → 共同時間比較／case-native 圖 | [31](../../scripts/31_compare_h_refinement_leading_edges.py)、[32](../../scripts/32_plot_h_refinement_refined_gll_leading_edges.py)；[refined_gll_leading_edge_artifacts](../../src/nek_post/refined_gll_leading_edge_artifacts.py) | 31 處理共同格點曲線比較，32 處理保存的 refined-GLL artifacts；兩者 metric 範圍不同。 |
| 原始速度快照 → native lambda_ci／重建 tensor 切片 | [33](../../scripts/33_compute_lambda_ci.py)、[34](../../scripts/34_plot_lambda_ci_slices.py)、[35](../../scripts/35_compute_lambda_ci_slice_series.py)；[lambda_ci_plotting](../../src/nek_post/lambda_ci_plotting.py)、[lambda_ci_series](../../src/nek_post/lambda_ci_series.py) | VTR／PVD／JSON／PNG／PDF／選用 NPZ；34 的 `--subdivisions-per-element 28` 對應 29 點，`--samples-per-element 32 --no-paraview` 支援分開的 Python-only 命名；不宣稱這就是歷史命令。 |
| 原始速度 → recovered omega_x → 保存標量 P7 切片 | [36](../../scripts/36_compute_omega_x.py)、[37](../../scripts/37_validate_omega_x_series.py)、[38](../../scripts/38_plot_omega_x_p7_midplane.py)、[39](../../scripts/39_compute_omega_x_p7_slice_series.py)；[omega_x_p7_series](../../src/nek_post/omega_x_p7_series.py) | 36 計算 native 場；37 驗證保存 volume 並建立 PVD；38/39 重建 scalar，不重算速度梯度。 |
| 原始速度 → native／GLL 重建三維渦場 | [40](../../scripts/40_export_vortex_3d.py)、[41](../../scripts/41_export_vortex_3d_series.py)；[vortex_3d_series](../../src/nek_post/vortex_3d_series.py) | binary VTU／JSON／PVD。manifest 與 sidecar 是 output/source 證據；signature 是模組指紋，不是已認證的歷史 Git revision。 |
| 保存 VTU → 序列化診斷 | [42](../../scripts/42_rewrite_vtu_diagnostic.py)；[vtu_compat](../../src/nek_post/vtu_compat.py) | 診斷副本與 subset mapping，與 production 渦場需分開。不能由工具存在推定結果樹中已有該實驗產物。 |
| 既有結果 → 盤點與發布 | [43](../../scripts/43_build_results_catalog.py)、[44](../../scripts/44_publish_results_catalog.py)；[catalog](../../src/nek_post/results_catalog.py) | 只整理 catalog，不驗證科學 payload。本階段不執行發布、不變更外部 README/CSV。 |

**UNKNOWN**：spectral_volume/N7 的歷史 entrypoint，及各歷史圖表的確切執行命令／revision。
若沒有對應 metadata，不由名稱指定 producer。

## 查詢 CLI 與再現工作的分界

從 repository root 可執行 `PYENV_VERSION=research312 python -B scripts/34_plot_lambda_ci_slices.py --help`
查核點數與 subdivision 選項；其他入口同樣使用 `--help`。
完整參數與輸出保護請讀[腳本索引](../script_index.md)及各方法文件。
這些是目前語法說明，不是歷史執行紀錄。本階段只讀 help，不啟動 production 計算。

需要將來再現時，先查 input/output mapping、物理時間、配置與空間預算，再選全新的
外部工作目錄。不要對現有 science root 使用 overwrite，亦不要修改腳本來迎合舊檔名。
