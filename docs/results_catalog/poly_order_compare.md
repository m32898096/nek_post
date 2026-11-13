# 多項式階數研究結果

由 [總覽](README.md) 進入已發布 results_index.csv，篩選 study=poly_order_compare，
再依 analysis、variant 與 entrypoint_file 選取分析。以下是既有產物的閱讀指南；
來源腳本的執行證據另見 [程式關聯表](producer_scripts.md)。

**VERIFIED FROM SOURCE CODE**：[cases.yaml](../../config/cases.yaml) 將 N5/N7/N9/N11
設定為 degree 5/7/9/11，reference_case=N11。
**VERIFIED FROM EXPLICIT METADATA**：
[t19p5 velocity table](/data/Nek5000_data/results/poly_order_compare/tables/velocity_error_t19p5.csv)
列出 N5/N7/N9 的 order=5/7/9 與 reference_case=N11。
這不代表每種分析都有四個階數。N7/P7 的 8 個原生 GLL 節點與 GLL10 視覺化取樣應分開。

| 分析類別 | 科學意義、案例及現有入口 | 證據與限制 |
| --- | --- | --- |
| figures、tables、reports | 濃度、速度、壓力及剖面之階數差異；[速度 L2 圖](/data/Nek5000_data/results/poly_order_compare/figures/velocity/t19p5/relative_L2_speed_vs_order_t19p5.png)、[multitime CSV](/data/Nek5000_data/results/poly_order_compare/tables/multitime_error_summary.csv)、[teacher report](/data/Nek5000_data/results/poly_order_compare/reports/teacher_error_summary.md)；PNG／CSV／MD | VERIFIED FROM EXPLICIT METADATA：上述誤差表以 N11 為數值參照。t05/t10/t15/t19p5 是選取標籤；[選取紀錄](/data/Nek5000_data/results/poly_order_compare/tables/comparison_set_t19p5_metadata.txt) 中 N5/7/9 index=79，N11 index=40；不是同幀比較，也不是精確解誤差。完整實際時間仍須回查來源控制。 |
| energy_budget_closure | [N5 CSV](/data/Nek5000_data/results/poly_order_compare/energy_budget_closure/N5_energy_budget_closure_timeseries.csv)、[N7 CSV](/data/Nek5000_data/results/poly_order_compare/energy_budget_closure/N7_energy_budget_closure_timeseries.csv)、[N9 CSV](/data/Nek5000_data/results/poly_order_compare/energy_budget_closure/N9_energy_budget_closure_timeseries.csv)、[summary](/data/Nek5000_data/results/poly_order_compare/energy_budget_closure/energy_budget_closure_summary.csv)、[closure figure](/data/Nek5000_data/results/poly_order_compare/energy_budget_closure/energy_closure_vs_time.png)；CSV／PNG | VERIFIED FROM EXPLICIT METADATA：summary 只有 N7，不能當作三案例共同摘要。三份時間序列表的案例依檔名標示，原始執行 revision UNKNOWN。圖表是後處理診斷，並非原始 energy_budget.dat。 |
| cantero_mean_front | [N7 CSV](/data/Nek5000_data/results/poly_order_compare/cantero_mean_front/N7/N7_cantero_mean_front_timeseries.csv)，同層有 N5/N9 | 空間 composite-GLL 等效高度門檻前緣，與固定平面抽取不同。VERIFIED FROM SOURCE CODE：script 24；既有執行腳本／revision UNKNOWN。 |
| cantero_front_reconstruction | [N7 comparison](/data/Nek5000_data/results/poly_order_compare/cantero_front_reconstruction/N7/N7_cantero_front_reconstruction_comparison.csv)；CSV／PNG | VERIFIED FROM SOURCE CODE：script 25 的正式入口限 N7；以 mean-front CSV 作時間梯度、平滑與積分重建後才比較 paper。不能把此目錄誤稱為 N5/N9 完整系列。 |
| cantero_re3450_multicase | [summary](/data/Nek5000_data/results/poly_order_compare/cantero_re3450_multicase/cantero_re3450_multicase_summary.csv)、[linear overlay](/data/Nek5000_data/results/poly_order_compare/cantero_re3450_multicase/cantero_re3450_multicase_linear_overlay.png)；CSV／PNG | VERIFIED FROM EXPLICIT METADATA：summary 列出 N5/N7/N9、front_source 與 Re3450 paper_source、平滑設定；是重建前緣的比較。與上一列 N7 產物可能有重疊，但來源流程不可只由檔名等同。 |
| front_kinematics | [summary](/data/Nek5000_data/results/poly_order_compare/front_kinematics/front_kinematics_summary.csv)、[velocity figure](/data/Nek5000_data/results/poly_order_compare/front_kinematics/front_velocity_smoothed_vs_time.png)；三案例時間序列表及 PNG | VERIFIED FROM SOURCE CODE：script 13 讀取外部 front_simple.dat，輸出原始位置、梯度速度、平滑速度及重建位置。這不證明 front_simple.dat 原本如何定義前緣。 |
| fig5a_paper_overlay、combined_xt_overlay | [paper summary](/data/Nek5000_data/results/poly_order_compare/fig5a_paper_overlay/fig5a_paper_overlay_summary.csv)、[processed summary](/data/Nek5000_data/results/poly_order_compare/combined_xt_overlay/processed_xt_paper_overlay_summary.csv)、[N7 Re3450/Re8950 子目錄](/data/Nek5000_data/results/poly_order_compare/combined_xt_overlay/N7_Re3450_Re8950_reconstructed/)；CSV／PNG | VERIFIED FROM SOURCE CODE：scripts 14/15 分別處理原始與 processed 外部前緣；17/18 處理 detected-front 路徑。須核對 paper_source 與案例 Reynolds number，不能把相同 N7 標籤當作同一物理案例。 |
| front_detection | [N7 summary](/data/Nek5000_data/results/poly_order_compare/front_detection/N7/N7_front_detection_summary.csv)、[N7_spectral summary](/data/Nek5000_data/results/poly_order_compare/front_detection/N7_spectral/N7_front_detection_summary.csv)、[Re8950 目錄](/data/Nek5000_data/results/poly_order_compare/front_detection/GC8950_N7_spectral/)；CSV／PNG、diagnostics | VERIFIED FROM SOURCE CODE：script 16 是從濃度場自動抽取的另一套前緣定義。N7 與 N7_spectral 的資料夾標籤提示 variant，具體方法應讀各表與原分析文件，不能自動選一個當作唯一正確結果。 |
| leading_edge_gll_refinement | [validation report](/data/Nek5000_data/results/poly_order_compare/leading_edge_gll_refinement/validation_report.md)、[full report](/data/Nek5000_data/results/poly_order_compare/leading_edge_gll_refinement/full_run_report.json)；JSON／CSV／PNG／PDF | VERIFIED FROM EXPLICIT METADATA：報告描述 N7/P7 在 8 或 10-node GLL 上的抽取，包含 rightmost-crossing 與 Moore-boundary；並未產生更高階解。行平均／中位數不是 spanwise quadrature 平均。 |
| leading_edge_right_most、leading_edge_benchmark、leading_edge_fix_test | [right-most 目錄](/data/Nek5000_data/results/poly_order_compare/leading_edge_right_most/N7/)、[benchmark CSV](/data/Nek5000_data/results/poly_order_compare/leading_edge_benchmark/N7/leading_edge_workers_benchmark.csv)、[fix-test 目錄](/data/Nek5000_data/results/poly_order_compare/leading_edge_fix_test/N7_moore_interpolated_full/) | CSV／圖與測試／效能紀錄需分開閱讀。benchmark 與 script 21 的功能相符，但目前程式預設為複數 leading_edge_benchmarks；舊目錄歷史 producer 不因此確定。見[分類說明](historical_results.md)。 |
| spectral_volume | [N7_GC0.pvd](/data/Nek5000_data/results/poly_order_compare/spectral_volume/N7/N7_GC0.pvd)；VTI／PVD | VERIFIED FROM EXPLICIT METADATA：PVD 記錄 81 個時間 0–20，連到既有 VTI。原始重建／取樣參數與 producer UNKNOWN；不能由 spectral_volume 字樣推定它等於 GLL10 vortex VTU。 |
| archive、energy_budget_closure_smoke | [歷史分類指南](historical_results.md) | 路徑標籤與實際紀錄共同支持有限用途分類；不作刪除、無效或淘汰判定。 |

## 能量資料的閱讀次序

**VERIFIED FROM SOURCE CODE**：[energy_budget.py](../../src/nek_post/energy_budget.py)
把輸入 E_k、E_p、E_total、epsilon 與時間保留，並計算
E_total − (E_k + E_p)、自首個觀測時間起的 trapezoidal integral(epsilon)、
energy_closure = E_total + cumulative_epsilon、相對 target 的 residual、
相對初值的 drift，以及 dE_total/dt + epsilon。
先查時間序列，再查 summary 的實際案例集合，最後讀圖。
這些是閉合診斷，不是新的求解器時間積分；target=12 是此 summary 的設定值。
外部原始 energy_budget.dat 不在本目錄盤點範圍內，來源與單位需另查。

所有表格與圖均為既有衍生產品。「CAPABLE / LIKELY SCRIPT」僅表示目前程式可支援該類功能；
沒有明確歷史執行證據時 producer 為 UNKNOWN。需要流向渦度與 swirling strength 的 native／
重建區別時，請改讀 [h-refinement 場產品指南](h_refinement.md)，避免把各種「spectral」標籤混用。
