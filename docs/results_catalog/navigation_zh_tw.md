# Nek5000 研究結果導覽

本目錄收錄 Nek5000 重力流研究的後處理結果，供流場、渦結構、前緣運動、
能量收支與紊流相關研究查找使用。根目錄為 `/data/Nek5000_data/results`；
以下連結皆相對於此根目錄。先依研究問題選擇入口，再閱讀該結果的控制檔與方法說明。

## 結果與目錄的範圍

既有科學檔案包含 native GLL 場的匯出、重建切片、衍生圖表、比較分析及歷史測試。
這些都是既有研究產物；原始 Nek5000 求解器快照通常位於結果根目錄之外，
實際來源以個別 sidecar 的 `source_file` 為準。圖表、重建網格與匯出的 VTK 檔案
各有用途，不能僅憑副檔名判定其數值精度或驗證程度。

`README.md` 與 [results_index.csv](results_index.csv) 是根目錄的導覽中繼資料。
CSV 是既有檔案的盤點快照，沒有重新計算科學量。`archive`、`diagnostics`、
`logs`、`smoke` 或 `fix_test` 類路徑可能保存歷史、診斷或小規模測試；
請保留這些產物並先查閱其用途。

## 目錄結構

| 研究群組 | 實際子目錄與用途 |
| --- | --- |
| [h_refinement](h_refinement/) | 網格細化研究。`omega_x`、`omega_x_highorder_slices`、`lambda_ci`、`vortex_3d_series` 提供場與切片；`field_comparison`、`convergence_analysis` 提供差異與收斂分析；`leading_edge` 提供不同前緣方法；`cantero_mean_front`、`cantero_re3450` 提供前緣及實驗比較。 |
| [poly_order_compare](poly_order_compare/) | 多項式階數比較。`figures`、`tables`、`reports`、`spectral_volume` 為圖表、報告及體積輸出；`energy_budget_closure` 為能量收支；`front_detection`、`front_kinematics`、`leading_edge_benchmark`、`leading_edge_gll_refinement`、`leading_edge_right_most` 為前緣分析；`cantero_mean_front`、`cantero_front_reconstruction`、`cantero_re3450_multicase`、`fig5a_paper_overlay`、`combined_xt_overlay` 為比較與疊圖；`archive`、`energy_budget_closure_smoke`、`leading_edge_fix_test` 保留歷史與測試。 |

N5、N7、N9、N7_H、N7_VH、N7_VVH 等名稱首先是目錄或表格標籤；
物理案例身分與來源版本仍需控制檔證據。CSV 的空白欄位表示未知，請勿由名稱補猜。

## 依研究問題快速導航

| 要找的結果 | 已確認存在的入口與相關控制檔 |
| --- | --- |
| N7_H 三維渦結構時間序列 | [81-frame vortex PVD](h_refinement/vortex_3d_series/gll10/case_N7_H/series/case_N7_H_vortex_3d_series.pvd)，配合 [manifest](h_refinement/vortex_3d_series/gll10/case_N7_H/series/manifest.json) 與 [processing summary](h_refinement/vortex_3d_series/gll10/case_N7_H/series/processing_summary.json)。PVD 的 81 個參照均存在，原記錄時間範圍為 0 至 20。 |
| frame 79 的三維場 | [GC0_f00079_gll10.vtu](h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.vtu) 與 [數值／來源說明](h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.json)。控制檔記錄 N7_H、P7 來源、每元素邊 10 個 GLL 取樣點，含 omega_x、lambda_ci，範圍為 x=0 至 17、y=0 至 1.5、z=0 至 1。 |
| native omega_x 體積 | [native volume series PVD](h_refinement/omega_x/case_N7_H/case_N7_H_omega_x_series.pvd)；[frame-79 metadata](h_refinement/omega_x/case_N7_H/case_N7_H/f00079/case_N7_H_f00079_omega_x.json) 記錄 native GLL、無插值。保留重複的 `case_N7_H/case_N7_H` 既有路徑。 |
| 重建 omega_x 切片 | [P7 slice series PVD](h_refinement/omega_x_highorder_slices/case_N7_H/case_N7_H_omega_x_p7_slice_series.pvd)、[frame-79 切片](h_refinement/omega_x_highorder_slices/case_N7_H/f00079/case_N7_H_f00079_omega_x_y0p75_xgt0_n29_p7.vtr) 與 [validation sidecar](h_refinement/omega_x_highorder_slices/case_N7_H/f00079/case_N7_H_f00079_omega_x_y0p75_xgt0_n29_validation.json)。此系列由已儲存的 native omega_x 標量作 P7 重建，平面為 y=0.75、x>0，每元素邊 29 個取樣點。 |
| native lambda_ci | 已確認存在的是 N7_H 的 [frame-79 native VTR](h_refinement/lambda_ci/case_N7_H/f00079/case_N7_H_f00079_lambda_ci.vtr)、[單幀 PVD](h_refinement/lambda_ci/case_N7_H/f00079/case_N7_H_f00079_lambda_ci.pvd) 與 [metadata](h_refinement/lambda_ci/case_N7_H/f00079/case_N7_H_f00079_lambda_ci.json)。metadata 記錄原生每方向 8 個節點，視覺化為相鄰 native GLL 節點間的線性體積單元。此入口只證實這一幀，不能據此宣稱有 81 幀 native lambda_ci 體積系列。 |
| 重建 lambda_ci 切片 | [y-midplane series PVD](h_refinement/lambda_ci/case_N7_H/case_N7_H_lambda_ci_y_midplane_series.pvd)、[x=15 series PVD](h_refinement/lambda_ci/case_N7_H/case_N7_H_lambda_ci_x15_series.pvd)、[frame-79 y 切片](h_refinement/lambda_ci/case_N7_H/f00079/slices/case_N7_H_f00079_lambda_ci_y_midplane.vtr) 與 [ParaView VTR metadata](h_refinement/lambda_ci/case_N7_H/f00079/plots/case_N7_H_f00079_lambda_ci_high_order_slices_paraview.json)。此 VTR 版本使用來源 GLL 基底及每元素邊 29 個均勻取樣點；[series summary](h_refinement/lambda_ci/case_N7_H/processing_summary.json) 記錄每元素邊 28 個 subdivisions，因此 28 + 1 = 29 個取樣點。frame-79 y 切片的 WholeExtent 尺寸為 3809 × 1 × 225。另有獨立的 [Python-only metadata](h_refinement/lambda_ci/case_N7_H/f00079/plots/case_N7_H_f00079_lambda_ci_high_order_slices.json) 與 [Python-only 圖](h_refinement/lambda_ci/case_N7_H/f00079/plots/case_N7_H_f00079_lambda_ci_y_midplane_xge0p0.png)，使用每元素邊 32 個均勻取樣點，其 metadata 不屬於上述 ParaView VTR。這兩種均勻取樣方式也與 GLL10 體積不同。 |
| H／VH／VVH 收斂分析 | [pairwise differences 圖](h_refinement/convergence_analysis/pairwise_differences.png)、[方向網格間距圖](h_refinement/convergence_analysis/directional_mesh_spacing.png)、[誤差表](h_refinement/convergence_analysis/finest_reference_errors.csv) 與 [analysis.json](h_refinement/convergence_analysis/analysis.json)。分析控制檔包含 N7_H、N7_VH、N7_VVH，以 N7_VVH 為可用的最細數值參照；這不代表已知精確解。 |
| N5／N7／N9 能量收支 | [N5 時間序列](poly_order_compare/energy_budget_closure/N5_energy_budget_closure_timeseries.csv)、[N7 時間序列](poly_order_compare/energy_budget_closure/N7_energy_budget_closure_timeseries.csv)、[N9 時間序列](poly_order_compare/energy_budget_closure/N9_energy_budget_closure_timeseries.csv)、[摘要表](poly_order_compare/energy_budget_closure/energy_budget_closure_summary.csv)、[能量閉合圖](poly_order_compare/energy_budget_closure/energy_closure_vs_time.png)。三份時間序列表依檔名區分案例；現有摘要表僅含 N7。閉合圖與各表的來源版本仍需另行查證。 |
| Cantero 實驗比較 | [多階數比較圖](poly_order_compare/cantero_re3450_multicase/cantero_re3450_multicase_linear_overlay.png) 與 [摘要表](poly_order_compare/cantero_re3450_multicase/cantero_re3450_multicase_summary.csv)；網格細化案例可另見 [H／VH／VVH 比較圖](h_refinement/cantero_re3450/cantero_re3450_multicase_linear_overlay.png) 與 [摘要表](h_refinement/cantero_re3450/cantero_re3450_multicase_summary.csv)。實驗條件及對齊方式需回查原分析來源。 |
| 前緣位置與方法差異 | [細化 GLL10 前緣圖目錄](h_refinement/leading_edge/refined_gll_nodes10/figures/) 及 [方法／案例報告](h_refinement/leading_edge/refined_gll_nodes10/figures/h_refinement_refined_gll_nodes10_report.json)，含 H／VH／VVH 的 Moore-boundary 與 rightmost-crossing；另有 [step6b 比較](h_refinement/leading_edge/step6b/comparison/summary.json)。 |
| 前緣運動學 | [front position 圖](poly_order_compare/front_kinematics/front_position_xt_vs_time.png)、[front velocity 圖](poly_order_compare/front_kinematics/front_velocity_smoothed_vs_time.png) 及 [摘要表](poly_order_compare/front_kinematics/front_kinematics_summary.csv)，同目錄保存 N5／N7／N9 時間序列表。 |
| 多項式階數比較圖 | [圖目錄](poly_order_compare/figures/) 分為 concentration、velocity、pressure、overlays 及 error_summary；例如 [t19p5 速度相對 L2 圖](poly_order_compare/figures/velocity/t19p5/relative_L2_speed_vs_order_t19p5.png)。檔名中的時間標籤僅供導航，跨案例比較仍需查核實際物理時間及座標。 |

## 科學慣例

- **P7 與 GLL10：** 上述三維 vortex sidecar 明確記錄來源為每元素邊 8 個節點、
  多項式次數 7；GLL10 是以既有 P7 多項式重建至每邊 10 個 GLL 節點。
  這增加視覺化取樣密度，沒有提升原始模擬解析度，也不能解讀成 P9 模擬。
- **幀編號與物理時間：** `f00079` 是檔案索引。上述 N7_H 控制檔記錄
  `time=19.50240408807`；完整 vortex PVD 的 81 個物理時間介於 0 與 20。
  保留原始時間值，跨網格比較應以實際物理時間配對。
- **omega_x：** 流向渦度，定義為 `∂w/∂y − ∂v/∂z`。
  上述 P7 omega_x 切片重建的是已投影的渦度標量。
- **lambda_ci：** swirling strength，由速度梯度張量特徵值的虛部大小定義，
  全實特徵值時為零。上述重建場流程先重建已投影的九個梯度分量，再求特徵值；
  不能用對 native lambda_ci 標量插值來取代此順序。
- **分類：** `artifact_class` 是路徑標籤推定的用途類別，`status` 是控制檔提供的
  完成／歧義資訊，`validation_level` 是盤點的驗證層級；三者必須分開解讀。

## 使用 CSV 目錄

以 Excel「資料 → 從文字／CSV」選取 [results_index.csv](results_index.csv)，
編碼選 UTF-8、分隔符號選逗號，將路徑及 ID 欄位保留為文字。
啟用篩選後，可依 `study`、`case_id`、`analysis` 篩選；在 `field_names`
搜尋 `omega_x` 或 `lambda_ci`。陣列欄位以 JSON 文字儲存。
`case_id` 空白的彙整表可能仍含多案例，請搭配 `variant` 與 `relative_path` 查找。

從 `entrypoint_file` 開啟 PVD、表格或圖，再看 `manifest_path`、`metadata_sources`
與 `notes`。路徑一律相對於本根目錄，移交他人時應保留整個相對結構。
`actual_time_start/end` 為讀到的控制檔物理時間範圍；`frame_count` 計算相異時間值，
不同切片在同一時間只計一次。缺少欄位證據時保持空白，不由檔名補值。

`status=reported_completed` 僅表示控制檔回報完成；`unknown` 表示缺乏足夠證據；
`incomplete_or_ambiguous` 表示存在不完整或歧義，例如宣告輸出缺失或參照格式不支援。
三者的 `validation_level` 均為 `inventory_only`，不代表科學結論已驗證。
請保留 `notes` 中缺失、格式、衝突與部分檢查範圍的診斷。

## 來源、驗證與維護

盤點只檢查路徑與有限的控制檔，不讀取大型科學 payload、不重算場值。
每個 JSON／PVD 控制檔最多讀取 1 MiB，每個資料群組最多檢查 64 個 JSON。
超大 relocation 紀錄、色盤陣列及不支援的歷史 schema 可能未獲完整查證；
部分 producer、Git revision、物理案例身分及跨目錄關係仍然未知。
歷史結果的來源資訊可能不完整。數學／獨立參照測試與正式 production Nek5000
執行檔的逐項比較是不同證據；目前沒有建立與實際 production 執行檔的精確等價性。

更新時先在結果樹之外產生新的預覽，檢查路徑、差異、診斷及科學檔案清單，
再由維護者審核。專案的 `docs/results_catalog/README.md` 說明實際 CLI。
目前發布工具僅建立新的根目錄檔案；任一目標已存在即拒絕，沒有原地覆寫更新模式。
更新現有已發布目錄需要另行審核及授權的替換程序；請勿用刪檔繞過碰撞保護。

- 使用目錄尋找現有結果，避免重複生成輸出。
- 不要手動搬移研究結果或更名 PVD 參照的幀檔案。
- 不要任意修改 scientific manifest、checkpoint、lock 或來源 signature。
- 保留 strict resume 的來源、檔案身分與完整性檢查。
- 根目錄僅有這兩個導覽檔案；不要在 production collection 或幀目錄加入導覽檔。
  同名但位於下層的既有檔案仍屬原始盤點範圍。
