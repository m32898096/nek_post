# Phase 3A：儲存庫架構與相容性稽核

本階段只有稽核、介面快照與後續設計；沒有移動程式、建立 wrapper、變更數值方法或開始 Phase 3B。研究者應先讀[架構稽核](phase3a_architecture_audit.md)，再讀[相容性矩陣](compatibility_matrix.md)及[分批計畫](refactor_plan.md)。紊流與渦流分析的科學定義仍以[數值方法索引](../methods/README.md)為準。

## 後續實作紀錄

| 已提交批次 | 責任與閱讀入口 |
| --- | --- |
| Phase 3B Batch 1 | [任務導覽](../guides/README.md)與[能源 CLI 相容性說明](../energy_budget_closure.md#入口與相容性) |
| Phase 3B Batch 2 | [script 08/13 的 CLI、API 與輸出契約](../guides/script_compatibility_contracts.md) |
| Phase 3B Batch 3 | [script 08 的 CSV helper 分工與保留介面](phase3b_batch3.md) |
| Phase 3C Batch 1 | [directional-GLL 測試 helper 歸屬](phase3c_batch1.md) |

原[分批計畫](refactor_plan.md)保存提案與風險分析；實際完成範圍請查各批紀錄。
本頁下方的來源清單、測試數與驗證日期均屬 Phase 3A 的固定基準。
數值方法與 production solver 證據限制請讀[方法索引](../methods/README.md)。
回到[repository README](../../README.md)或[任務導覽](../guides/README.md)。

## 基準與證據層級

- 基準 `dev`：`0048d9e0c070133403f2789d3fd27186a37fbbe9`；新分支 `feature/repository-refactor`。
- 開始時工作樹乾淨；live `origin/dev` 與本機 `dev` 相同。已驗證 Phase 1 merge `017cd24880f5b398c0df7b8f091360ab43ba849c`、Phase 2A `26e8e471c9633e0d80916378ebd7cb69766f8026`、2B `f18f94afc66bee4d0e34d5fac81a856d1135306d`、2C `99a4828d15f2e724300b05e3a2801bd70c51dd21` 均為基準祖先，且基準是已公開的 Phase 2 merge。
- 開始時本機與 live `origin/master` 都是 `72f08bdb64dd45785450189da677031f44b15439`；沒有未完成 merge。此階段不更動這些分支，也不提交或推送。
- **CONFIRMED FROM SOURCE**：可由本次固定版本原始碼、Git 或受限控制檔讀取直接確認。
- **SUPPORTED BY TESTS**：現有測試在本機環境通過；不表示所有外部使用情境皆已驗證。
- **INFERRED**：依來源結構提出的責任分類、風險或未來設計，尚未實作。
- **UNVERIFIED**：沒有足夠證據，例如外部研究腳本、排程、舊 pickle、production solver executable 的完全對等性。

## 可查詢的逐檔基準

| 檔案 | 範圍與用途 |
| --- | --- |
| [repository_inventory.json](repository_inventory.json) | 280 個既有檔案：46 scripts、98 source、101 tests（100 Python 與 1 Fortran driver）、35 docs。逐檔路徑、內容 SHA-256、責任、函式/類別與原始簽章、匯入、CLI 參數/default 表達式、測試/文件引用、路徑及輸出標記、風險、五組 signature 成員；另有模組依賴圖。 |
| [cli_contracts.json](cli_contracts.json) | 46 個腳本完整 `--help` stdout、stderr、return code；全部為 0。這是目前環境、目前 config 的快照。 |
| [source_fingerprints.json](source_fingerprints.json) | 五個工作流程的完整 basename → SHA-256 字典，以及已發表 vortex frame 79、manifest、summary 的 signature 比對。 |

JSON 是可供獨立審查的文件附件，沒有加入執行時依賴。inventory 排除本目錄與 `__pycache__`；根 [README](../../README.md)、[pyproject.toml](../../pyproject.toml)、[requirements.txt](../../requirements.txt) 與 config 另由本文及稽核說明。檔案雜湊不是科學資料內容雜湊。

inventory 的 `imports` 包含函式內、條件式匯入；圖中的 cycle 不等於 eager import 失敗。`direct_production_imports` 是 literal import 關係；`referenced_by` 是字面路徑、檔名或模組名稱搜尋；`associated_tests` 結合兩者，不是完整呼叫圖或覆蓋率。沒有 docstring 的測試責任由實際 import/test 名稱推定，並標記 **INFERRED**。族群層級的輸出說明也明確標記推定，精確行為須回到該檔案的定義與路徑標記。動態計算的 import、外部 consumer 不會因為搜尋沒有命中而被判定不存在。類別欄位、繼承及 decorator 的全部 runtime 行為須讀原始碼；AST 函式簽章不是完整 Python ABI 規格。

## 本次驗證

**SUPPORTED BY TESTS**：使用 `PYENV_VERSION=research312`、`python -B`、`PYTHONDONTWRITEBYTECODE=1`。所有 pytest 加上 `-q -p no:cacheprovider`，並以 `--basetemp /tmp/phase3a-audit/...` 隔離產物。

| 檢查 | 結果 |
| --- | --- |
| 完整現有 pytest | **2,786 passed，111.10 s**；沒有 skip。 |
| 數值/reference、strict resume/provenance、catalog focused suite | **1,007 passed，47.79 s**；是完整 suite 的子集，不能相加為不同測試總數。 |
| 全部 script `--help` | **46/46 passed**；沒有執行分析或發表命令。 |
| production module imports | **98/98 passed**；包含 package `__init__`，不保證任何可選功能或不同匯入順序。 |
| Python AST syntax | **244/244 passed**；46 scripts + 98 source + 100 tests。 |
| Markdown local targets | **361/361 存在**：313 個既有 targets + 48 個本次新增 targets；publication template 以 scientific root 解析。未把外網或 renderer anchor 當成已驗證。 |
| JSON / 基準一致性 | 3 個新 JSON 可解析；280 個 inventory 檔案 hash、五組 source maps、CLI entrypoint 集合與基準一致。 |
| 範圍 / whitespace | 僅本目錄 7 個新文件未追蹤、未 staged；既有 tracked/index diff 空白。`git diff --check`、cached check、新檔逐一 `git diff --no-index --check /dev/null FILE` 均通過。 |

focused suite：`test_independent_numerical_reference.py`、`test_gll_differentiation.py`、`test_spectral_derivatives.py`、`test_l2_projection.py`、`test_scalar_projection.py`、`test_vorticity.py`、`test_swirling_strength.py`、`test_vortex_3d_reconstruction.py`、`test_vortex_3d_series.py`、`test_vortex_3d_artifact_integrity.py`、`test_vortex_3d_regressions.py`、`test_omega_x_workflow.py`、`test_omega_x_p7_series.py`、`test_results_catalog.py`、`test_results_catalog_publication.py`、`test_results_catalog_guides.py`（皆在 `tests/`）。

文件連結、JSON、逐檔基準、科學目錄安全比對及 whitespace 結果見[驗證邊界](phase3a_architecture_audit.md#驗證與未解問題)。不執行 production resume；控制檔 signature 一致僅證明其中一項條件，不代表 81-frame 全面重驗，也不代表與 production Nek5000 executable 完全相等。
