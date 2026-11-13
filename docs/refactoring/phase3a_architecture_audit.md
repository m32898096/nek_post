# Phase 3A：架構稽核與保留邊界

基準、逐檔資料及本次測試結果見[索引](README.md)。以下「已確認」均指 **CONFIRMED FROM SOURCE**；重組方案均是 **INFERRED**，不授權執行重組。

## 目前架構

已確認：[pyproject.toml](../../pyproject.toml) 使用 setuptools `src/` layout、Python ≥ 3.12，沒有 `[project.scripts]` entrypoint。`nek_post` 的 [__init__.py](../../src/nek_post/__init__.py) 只有 docstring；現有介面主要是直接 import 子模組與 `python scripts/XX_name.py`。根 [README](../../README.md) 說明 editable install。移動腳本時不能假設 repo root 自動成為 `src` 的 import path。

[requirements.txt](../../requirements.txt) 是完整環境的版本清單，含 NumPy、SciPy、Pymech、Matplotlib、VTK/PyVista、PyYAML、Numba 等；它不是只含直接依賴的最小 runtime contract。逐檔第三方 import root 見 inventory；標準庫已排除，optional/lazy import 仍保留。此階段不調整 packaging 或依賴版本。

| 現有模組族群（`src/nek_post/`） | 責任與已確認的主要關係 | 保留建議 |
| --- | --- | --- |
| `io_nek`, `fields`, `config`, `paths` | Nek header/field 讀取、field accessor、YAML 與結果路徑；供各 workflow 使用。 | 保留公開 import；path depth 與 config 預設不可悄悄更換。 |
| `gll`, `spectral_derivatives`, `l2_projection`, `vorticity`, `swirling_strength` | GLL/微分、實體 Jacobian/gradient、mass-lumped recovery、signed scalar curl、投影後 tensor 的 eigenvalues。 | 不移動、不抽取、不調整 import；維持 Phase 1 方法與來源雜湊。 |
| `spectral_interpolation`, `spectral_horizontal_slice`, `refined_gll_horizontal_slice`, `gll_refinement`, `gll_directional_integration`, `gll_directional_workflow` | element-aware sampling、實體平面、既有多項式重取樣、方向積分/NPZ。 | 相同名稱的 interpolation 並非相同方法；inverse map、固定網格與 element-local sampling 不可合併。 |
| `lambda_ci_workflow/export/plotting/series/slice_export`, `rectilinear_export` | 原生 VTR、dense tensor slices、圖與 PVD。依賴 derivative → projected gradient → swirling 的順序。 | 原生、Python 圖與 ParaView dense VTR 的 variant 必須分開。 |
| `omega_x_workflow/series_pvd/p7_slice/p7_series` | 原生 curl recovery、saved-volume validation、已投影 scalar 的 P7 sampling。 | saved scalar 不重新計算 curl；resume 例外不是一般遷移政策。 |
| `vortex_3d`, `vortex_3d_workflow/series/reconstruction/selection/artifacts`, `hex_validation`, `vtu_compat` | 二進位 VTU、volume/crop、P7、frame/series 安全驗證；`vtu_compat` 是獨立 serialization 診斷。 | strict source identity 與實際 artifact identity 都需保留。 |
| `comparison*`, `interpolation`, `metrics`, `slicing`, `plotting` | p-study/common-grid slice、時間選擇、誤差與報表。 | 已有 kernel/IO/reporting 分界，不為目錄整齊而搬移。 |
| `front_detection*`, `fixed_grid_interpolation`, `front_compare`, `front_io`, `fig5a_detected_front*` | 固定網格濃度、追蹤、cache、平行 preprocessing、reference comparison、診斷圖。 | cache、tracking 與 visualization-only reconstruction 不可互換。 |
| `leading_edge_*`, `leading_edge_methods/*`, `refined_gll_leading_edge*` | threshold/edge extraction、方法 dispatch、Moore tracing、workflow、CSV、workers、benchmarks。 | `rightmost_crossing` adapter 有相容目的；不能視為無用重複。 |
| `cantero_*`, `reconstructed_xt_comparison*`, `front_kinematics*` | equivalent-height quadrature、mean front、時間 reconstruction、paper overlay、速度/報表。 | 科學 front 定義與不同 Re/case 的配對條件不可合併。 |
| `energy_budget*` | 純數值 closure、輸入/CSV/terminal table、plots 三層。 | 數值層不動；只有小型 reporting extraction 值得後續評估。 |
| `h_refinement*` | mesh inventory、實體時間選擇、共同座標場比較、條件式 convergence、front/phase 分析。 | 不用 element index 或 filename 取代時間/實體座標證據。 |
| `results_catalog`, `results_catalog_publication` | bounded read-only inventory、dataset identity、schema CSV、獨占建立兩個 catalog root 檔。 | 現有分界清楚；Phase 2 修正全部保留。 |

每個實際檔案（含上述表未逐一展開的子模組）的責任、函式、依賴及測試對應均在 [inventory](repository_inventory.json)。`module_graph` 邊表示來源中存在 import，不代表所有呼叫。`direct_import_consumers` 與 `associated_tests` 可反查 source 的既有使用者。

## 腳本與 CLI

已確認共有 **46** 個 Python entrypoints：00–44 各一個，加 `plot_refined_gll_leading_edge_evolution.py`。完整逐檔用途、參數與 default 原始表達式在 inventory；實際完整 help 在 [CLI 快照](cli_contracts.json)。只有 `--help` 被執行，未執行科學計算。分組用途可作為未來導覽分類，**不是搬移清單**：

| 編號/舊位置 | 科學用途與輸入/輸出關係 | 主要相容風險 |
| --- | --- | --- |
| `scripts/00`–`01` | field discovery 與 header/field probe。 | YAML/case/data-root、repo-depth 假設。 |
| `scripts/02`–`11` | slice → polynomial comparison → summary、multi-time orchestration/overlay/table。 | 05/07 以 subprocess 呼叫 02/03/04；舊檔名與 cwd 有實際意義。 |
| `scripts/12`–`15` | energy DAT closure、front DAT kinematics、paper/reconstructed front overlays。 | 預設資料根、CSV/figure 命名、覆寫與 terminal reporting。 |
| `scripts/16`–`21` | detected front、four-way overlay、leading-edge compute/plot/benchmark。 | script 19 也被 benchmark 呼叫；tests 使用 file-based import 與 monkeypatch。 |
| `scripts/22`–`26` | GLL directional NPZ、Cantero height/mean front、temporal reconstruction、多 case overlay。 | dimension/direction/definition、先 reconstruction 後 comparison 的順序。 |
| `scripts/27`–`32` | h-study inventory、common-plane/physical-time comparison、convergence、saved edges。 | h-study config/output root 不可回退到 p-study。 |
| `scripts/33`–`35` | lambda_ci native snapshot、dense slices、resumable series。 | Python-only 32 nodes 與 ParaView 29 nodes/28 subdivisions 不混用。 |
| `scripts/36`–`39` | native omega_x、saved series validation、P7 plane/series。 | 禁止把 saved-scalar 重建改成新的 curl/recovery；exact provenance。 |
| `scripts/40`–`41` | one-frame/series native 或 P7 binary VTU。 | immutable config、crop、source signature、locks、strict resume。 |
| `scripts/42` | 既有 VTU 的 serialization-only 診斷。 | 不是 production writer，viewer acceptance 仍需另證。 |
| `scripts/43`–`44` | bounded catalog 與 preview/explicit publication。 | readonly scan 與 root-only publication 分開；不可自動覆寫已發表檔。 |
| unnumbered plot script | saved refined-GLL leading-edge CSV 的 legacy-style plot。 | `read_run` 也可被 import；不是只有命令列入口。 |

上述縮寫對應精確檔名以 inventory 的 `path` 為準。[05](../../scripts/05_run_t19p5_pipeline.py) 與 [07](../../scripts/07_run_multitime_pipeline.py) 有明確子程序路徑；[leading_edge_benchmark.py](../../src/nek_post/leading_edge_benchmark.py) 第 269 行建立 `scripts/19_compute_leading_edge_evolution.py`。多數 script 使用 `Path(__file__).resolve().parents[1]` 找 repo/config。只搬到 `scripts/group/` 會破壞 depth。`--help` 成功只測 parser/import/config-read 路徑，沒有驗證所有分析參數組合。

**UNVERIFIED**：repo 外的 shell/batch、notebook、教師命令與排程沒有完整清單。checkout 未找到 `.sh`/`.bash` 工作腳本；不能據此斷言外部無 caller。歷史命令與 producer-script 欄位是追溯資訊，不能在整理文件時用新路徑回填歷史。保留所有舊 entrypoint，比一次新增 46 個 wrapper 更容易維護；只在批准的 extraction 才增加單一 wrapper。

## 分界、重複候選及 runtime 風險

**CONFIRMED FROM SOURCE**：大型模組包括 `leading_edge_benchmark.py`（1,248 行）、`front_detection_cache.py`（1,215）、`gll_directional_integration.py`（1,001）、`spectral_interpolation.py`（871）、`leading_edge_methods/moore_boundary.py`（804）。長度是調查線索，不是分拆理由。benchmark 同時含資源/子程序/報告，cache 含驗證/序列化/建構，energy IO 亦含報表；抽取前需先固定各介面及 failure 行為。

**INFERRED** 的重複候選：frame-index parsing、atomic JSON/no-replace publishing、plot/report formatting、similar NPZ/CSV validation。不同 workflow 有不同 schema、legacy token、鎖與 overwrite 規則；不能用 generic helper 直接取代。`interpolation.py`、`fixed_grid_interpolation.py`、`spectral_interpolation.py` 看似相近但對應不同幾何/數學契約。數字 script、`*_io`、`*_workflow`、`*_artifacts`、`*_export` 的命名反映歷史用途，不假設相同 suffix 就有相同抽象層。

**CONFIRMED FROM SOURCE**：保留 lazy imports 的 static graph 有三組 strongly connected components：

- `fig5a_detected_front` ↔ `fig5a_detected_front_plotting`（前者第 289 行延遲匯入 plotting）。
- `lambda_ci_plotting` ↔ `lambda_ci_slice_export`（前者第 336 行，後者第 46 行是函式內匯入）。
- `vortex_3d`、`vortex_3d_artifacts`、`vortex_3d_reconstruction`、`vortex_3d_selection`（含 artifacts 第 174 行延遲匯入 selection）。

目前 98 module import 與完整 suite 通過，沒有證據顯示 eager circular-import 故障；將 lazy import 拉至檔頭反而可能引入故障。plot modules 的 backend 設定、optional imports 與 multiprocessing worker 的 importability 都需維持。`_positive_integer`、`_rectilinear_geometry`、`_file_hash` 等 underscore helper 被其他 production 模組使用；不可按名稱推斷為可任意移除的 private API。

[test_leading_edge_sampling.py](../../tests/test_leading_edge_sampling.py) 第 88 行測試 pickle roundtrip、第 99 行使用 spawn process pool。Python object 的 module/qualname 必須可載入；scientific NPZ 的 `allow_pickle=False` 不代表 worker pickle 相容性已無關。沒有在 bounded JSON 中找到 qualified name，不能證明外部 serialized object 不存在。長期舊 pickle 消費者 **UNVERIFIED**。

## 測試架構

已確認：100 Python test modules + `tests/reference/nek_gll_driver.f90`。每檔 `direct_production_imports`、test definitions、literal script 引用及 `cross_test_imports` 都已列於 inventory；不以命名取代實際 import 關係。

| 類別 | 現有代表與保護的契約 |
| --- | --- |
| 純 unit/kernel | `test_metrics`、`test_energy_budget`、`test_leading_edge_thresholds`、`test_spectral_derivatives`。 |
| independent numerical/reference | `test_independent_numerical_reference`、`test_gll_differentiation`、`test_scalar_projection`、`test_vorticity`；analytic/explicit connectivity 期望保持獨立。 |
| CLI/file import | `test_compute_*_cli`、`test_integrate_gll_field_cli`、`test_benchmark_leading_edge_workers_cli` 及 workflow 中的 CLI checks。 |
| workflow/writer integration | `test_lambda_ci_workflow`、`test_omega_x_workflow`、`test_vortex_3d*`；小型 synthetic Nek/VTK roundtrip。 |
| resume/provenance/safety | `test_vortex_3d_series`、`test_vortex_3d_artifact_integrity`、`test_omega_x_p7_series`、`test_front_detection_cache` 等。 |
| process/cache/plot | `test_leading_edge_sampling/parallel/benchmark`、`test_front_detection_parallel/cache`、plot tests。 |
| catalog/publication/docs | `test_results_catalog`、`test_results_catalog_publication`、`test_results_catalog_guides`。 |

31 test modules 含 `__file__`，15 個 import sibling `test_*` helper；例如 `test_vortex_3d_series` 使用 `test_lambda_ci_workflow.synthetic_file`，artifact-integrity 又使用 series helpers。這是 collection/path coupling，不是已證實的執行順序故障。沒有做 randomized order 測試，順序完全獨立性 **UNVERIFIED**。任意加入 `tests/__init__.py`、切換 pytest import mode 或搬成 package 可能改變既有 import/monkeypatch identity。

外部依賴已確認：[independent reference test](../../tests/test_independent_numerical_reference.py) 第 71–86 行可選擇讀取固定 local Nek `speclib.f`、核對 Phase 1 hash，用 gfortran 在 pytest temp 編譯 standalone GLL driver，不啟動 solver。第 79 行 evidence 路徑也依 repo cwd。`test_results_catalog_guides.py` 第 27 行開始可選擇唯讀 bounded production controls；缺檔會 skip。此次沒有 skip，不代表可攜環境必有這兩種資源。其他 catalog/workflow tests 使用 synthetic temporary fixtures；本階段不將私有資料設為新必備測試依賴。

建議邏輯分組：`tests/numerics/`、`tests/workflows/`、`tests/cli/`、`tests/catalog/`、`tests/reporting/`、`tests/reference/`。這是導覽目標，不是立即搬檔批准。先抽出明確 fixture contract、驗證單檔獨立 collection、維持 driver 相對路徑及同名 import，再提出小批搬移。下一階段只考慮無跨 test import/實體路徑的 reporting/energy unit 小批，見計畫。

## 文件架構

已確認：入口為根 README、[script index](../script_index.md)、[workflow](../workflow.md)、[methods index](../methods/README.md)、[results catalog](../results_catalog/README.md) 與研究者導覽。35 個既有 docs 全部列於 inventory。方法來源、scientific workflow、實作歷史與使用範例目前混放；例如 945 行的 [vortex guide](../vortex_3d_binary_vtu.md) 同時保存歷史批次/修正與現行方法。

已確認的導覽陳舊文字：根 README 第 52 行仍稱 tests 是 metrics 的 minimal tests；第 57 行仍以 proposed independent validation 描述 Phase 1 入口。它們不精確反映現有 suite/Phase 1 完成狀態，但本階段不重寫既有文件。沒有把「未使用/過時」等同可刪除：`multitime_plan.md`、`t19p5_summary.md`、舊方法比較、舊 CLI 命令均保留。

**INFERRED** 目標：根 README 做任務導覽；`docs/methods/` 保留數學/solver evidence；`docs/results_catalog/` 保留產品與解讀；新增 `docs/guides/README.md` 僅連到現有 workflow/CLI guides；`docs/refactoring/` 保存決策。既有 workflow 文檔先不搬。相同 L2/omega/lambda 說明可透過連結指向 canonical methods，但不能削掉歷史背景或改寫歷史命令。搬文件會改變相對連結解析基準；必須保留舊入口及逐條驗證 target 的產品語意，僅檔案存在不夠。

## 驗證與未解問題

**SUPPORTED BY TESTS**：完整 2,786 passed；focused 1,007 passed；46 help、98 imports、244 AST 全通過，詳[索引](README.md)。新增文件及 JSON 連結/基準一致性以本次唯讀檢查驗證；既有文件的 historical code examples 不當成目前 API guarantee。Markdown 檢查範圍為 inline local file/directory targets，不是外網、所有 renderer anchor 或 prose 中的命令執行。

**CONFIRMED FROM SOURCE / 本機唯讀觀測**：Phase 3A 前後 external inventory 比較路徑、size、mtime_ns；原科學檔 2,556 個共 40,760,621,218 bytes，另有已發表 README/CSV，physical file 共 2,558。小型 JSON/CSV/PVD 與 README 另核對 SHA-256，沒有雜湊/解碼約 40 GB payload。亦與保留的 Phase 2 integration 前快照比對，以涵蓋 help 檢查開始前的狀態。此證據不是大檔內容逐 byte 等同性的獨立證明；符合本階段 bounded-read 安全要求。沒有 production locks、checkpoint 或 metadata 寫入；沒有 production resume、simulation 或科學輸出重建。

尚未解決、不影響本次 audit 文件完成的問題：

1. **UNVERIFIED**：repo 外 imports、命令、排程與序列化 consumer，須由使用者/後續調查補充。舊路徑預設保留。
2. **UNVERIFIED**：搬移後全 import order、spawn/pickle 與依賴 optional features；本次沒搬任何檔。
3. **UNVERIFIED**：81 frames 所有 source/VTU/array checksums；只讀小型 controls 與 frame 79 signature。不因 signature 相同就宣稱完整 strict resume 通過。
4. **UNVERIFIED**：matched production Nek5000 executable、compiler/runtime、BC/connectivity 與 reference payload 的完全對等性；Phase 1 [限制](../methods/nek5000_provenance.md) 持續有效。
5. **INFERRED**：通用 publishing/cache helper 的收益可能小於 provenance 維護成本；在單一 unsigned extraction 證明收益前延後。
