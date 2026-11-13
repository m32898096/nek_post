# Phase 3B Batch 3 儲存表格彙整邏輯抽取

本批基準為 `ed8fe68a6f06a81904fb6feada4c9c9a75507743`。
依[已接受計畫](refactor_plan.md)的小批次 CLI／IO 分工原則，選擇已有 Batch 2
契約的 [script 08](../../scripts/08_collect_multitime_error_summary.py)。

## 責任與移動範圍

原 script 08 共 252 行，混合 argparse、config 選擇、CSV schema、欄位映射、驗證、排序與寫檔。
其他程式要使用 CSV 彙整功能時，需要以檔案位置載入 numbered script。
新 [nek_post.multitime_summary_io](../../src/nek_post/multitime_summary_io.py) 讓一般 package import
可直接使用這組已存表格 IO 功能。

| 原 script 08 符號 | 新模組中的實作 |
| --- | --- |
| `FIELD_ORDER`, `FIELD_MAPPINGS`, `COMMON_INPUT_COLUMNS`, `OUTPUT_COLUMNS` | 同名 schema/constants |
| `_require_columns` | 同名內部欄位驗證函式 |
| `_collect_file` | `collect_error_file` |
| `_sort_key` | `error_row_sort_key` |
| `_write_summary` | `write_error_summary` |

script 08 明確匯入舊名稱作為 aliases，保留 helper 的直接呼叫簽章及 constants 的共享物件。
argparse、config/repo path、input filename、`main()`、錯誤捕捉與 terminal output 仍由舊入口協調。
新模組只依賴標準庫 `csv` 與 `pathlib`，import 時無資料讀寫。
本次沒有抽取能源 CLI，也沒有建立 Phase 3C 的新 CLI package。

## 基準與驗證

**VERIFIED AGAINST COMMITTED BASELINE** 的來源是修改前的 `ed8fe68` script 08，SHA-256 為
`39f234939faae9bd82b8544708ff9b2cd793445f2b861d474cddd2efb44d5344`。
修改前在 repo 外保存了 source、helper signatures、固定 `COLUMNS=80` 的 help，以及小型
velocity CSV 彙整的 bytes、stdout/stderr/exit status。
新測試把這個既有 CSV 與 terminal output 明確列為期望值，路徑部分由暫存目錄代入。
原 Phase 3A help capture 也與修改前 stdout 一致，維持 exact comparison。

[相容性測試](../../tests/test_script_compatibility_contracts.py) 保留 Batch 2 的獨立欄位映射、
排序、缺檔、malformed CSV 與覆寫契約，新增 legacy aliases、call parameters、標準庫模組 import
及 pre-extraction CSV/help 的回歸。script 13 與 script 12 契約持續執行。

本次搬移的函式 body 與 constants 由原檔直接抽取，沒有改動指標計算。
彙整器仍複製既有 metric strings，並依既有規則排序，summary 仍直接覆寫。
數值等價、來源 hash、import path、執行結果與 strict resume 是分別驗證的契約。
兩個相同輸出不能代替 source fingerprint 核對。

**VERIFIED BY SOURCE INSPECTION** 表示查核過五組明列 fingerprint inventories，script 08
及新模組均非成員，也未新增進入 protected workflows 的 import。
所有 protected source 的內容與 sibling paths 保留，既有 strict-resume 驗證仍適用。
不執行 production resume，也不回填歷史來源紀錄。

**UNVERIFIED** 的範圍包括 repo 外對 script private globals 的重新綁定，以及把 private helpers
以原 module-qualified name 序列化的用途。移動後函式的 `__module__` 會指向新模組。
目前 checkout 只有既有 compatibility tests 直接載入 script 08，未找到這些外部依賴。
本批保留既有 symbol names、參數及直接呼叫路徑，沒有承諾任意 private-global monkeypatch 相容性。
歷史命令與 producer references 保留原 numbered path。

## 執行回歸

```bash
COLUMNS=80 PYENV_VERSION=research312 python -B -m pytest -q -p no:cacheprovider tests/test_script_compatibility_contracts.py tests/test_energy_budget_cli_compatibility.py
COLUMNS=120 PYENV_VERSION=research312 python -B -m pytest -q -p no:cacheprovider tests/test_script_compatibility_contracts.py tests/test_energy_budget_cli_compatibility.py
PYENV_VERSION=research312 python -B -m pytest -q -p no:cacheprovider
```

所有 executable fixtures 在 pytest 暫存區內建立 config/input/output，production results 只做
bounded inventory/control 檢查。更高風險的 signed numerical modules 搬移、publishing helper
合併與 workflow extraction 維持延後。下一步由獨立 review 評估此單一分界，再決定是否需要
另一個 scripts batch，或另行授權 Phase 3C。
