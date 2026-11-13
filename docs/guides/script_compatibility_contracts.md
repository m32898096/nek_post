# Phase 3B Batch 2：既有腳本相容性契約

基準為 `a4e671f37cb9b90cf7609c04e9739215db82e3ea`，Phase 3A 為
`83e16004a1773852f713a51228f072f39c83f6a7`。依[已接受計畫](../refactoring/refactor_plan.md)
的 3B.2 先建立測試，不搬移程式、不開始 Phase 3C。

## 選擇與原有涵蓋

| 固定入口 | 選擇理由與依賴 | 先前測試與本批新增契約 |
| --- | --- | --- |
| [08_collect_multitime_error_summary.py](../../scripts/08_collect_multitime_error_summary.py) | 彙整已存誤差 CSV，使用 `nek_post.config.load_project_config` 與 `ProjectPaths`；預設依腳本位置找 config，整理入口時易受影響。 | 原測試沒有直接引用此入口。本批測 CLI、預設 config/table 路徑、14-column schema、三種 field 指標映射、時間/field/order 排序、reference-case fallback、缺檔與 malformed CSV 診斷。 |
| [13_front_kinematics.py](../../scripts/13_front_kinematics.py) | DAT → 速度/重建 → CSV/圖的流程協調；使用 `front_io`、`front_kinematics`、`front_kinematics_io/reporting/plotting`、`paths`。 | 既有 `test_front_kinematics*` 保護 kernel、IO、報表及圖，但沒有 CLI 執行測試。本批補 parser defaults/types/choices、import identity/call parameters、實際 subprocess、default/explicit paths、輸出 schema、缺檔與 collision。 |

兩個 script 不在 Phase 3A 五組 source signatures 內；本批不修改它們，也不改任何 signed module。
能源 script 12 的 Batch 1 契約保留，不重複實作。

## 參考值與執行安全

[新測試](../../tests/test_script_compatibility_contracts.py) 的 schema、參數與 overwrite 行為
來自上述 committed source inspection（**VERIFIED AGAINST COMMITTED BASELINE**）。
沒有新增 help snapshot；檢查已知 flags、成功 help 與 unknown-option exit 2。
每個 subprocess 使用 copied environment，明確固定 child `COLUMNS=80` 及暫存
`MPLCONFIGDIR`；不改 global `os.environ`。help 描述文字與排版未全面鎖定。

所有有輸出行為的執行使用暫存 `scripts/` + `config/` 結構，逐 byte 複製目前 entrypoint，
不改腳本內容，配合暫存 YAML/DAT/CSV。這讓無 input-directory flag 的 script 08
不接觸 production tables；也測試 `__file__` 相對 config 與不同 cwd 的行為。
原 repo 的兩個固定入口另外直接執行 `--help` 與 invalid argument。
此模式保護入口邏輯和 repo layout convention，不是完整新環境安裝測試。

- 前緣參考為獨立解析式 `x(t)=1+2t`，輸入刻意使用未排序、不等距的物理時間
  `12,3,7,4`；預期速度 2、積分重建原位置、誤差 0，容差 `atol=1e-12, rtol=0`
  僅容納 gradient/浮點 roundoff（**VERIFIED BY INDEPENDENT REFERENCE**）。
- 彙整器輸入是手寫小型 legacy CSV；預期欄位映射、字串值及排序直接列明，
  不重新計算誤差量，不由受測函式產生預期值。
- script 13 拒絕既有輸出，只有 `--overwrite` 才可取代；script 08 依既有行為直接
  覆寫 summary，沒有 no-overwrite 保護。本批明確保留此差異。
- 同次執行前後 output bytes 相等只用於重跑/覆寫保留檢查，屬 **SELF-CONSISTENCY ONLY**，
  不是獨立數值參考。各 input bytes 保留；沒有重建既有 golden。

## 公開介面、來源與限制

測試確認 script 13 仍匯入原 `compute_kinematics`、`read_front_simple_dat`，
保留現有參數名稱；parser 的 private helper 只用於檢查 CLI defaults，不新增公開 API。
各源檔路徑與內容由五組 Phase 3A signature 快照和獨立 SHA-256 檢查固定，
並核對現行 `_signature()` 輸出。這是 **VERIFIED AGAINST COMMITTED BASELINE** 的
source-integrity guard，不等同於 import/runtime/數值等價或 production strict-resume 通過。
不得為通過測試而更新來源 hash、放寬 resume 或回填歷史 metadata。

**UNVERIFIED**：這些 CLI 測試不涵蓋完整 Savitzky–Golay/plot 路由、所有資料異常組合、
所有外部 caller 或全 CLI help 描述；既有 kernel/plot tests 持續保留。
沒有執行 production resume 或讀寫 81-frame payload。catalog grouping、25-column schema、
ID 與 publication 未修改，仍由既有 catalog tests 保護；本批 14-column aggregation CSV
是不同資料產品，不是 catalog schema。

## 執行

在根目錄執行；所有產物放在 pytest 暫存區（可用 repo 外 `--basetemp`）：

```bash
COLUMNS=80 PYENV_VERSION=research312 python -B -m pytest -q -p no:cacheprovider tests/test_script_compatibility_contracts.py tests/test_energy_budget_cli_compatibility.py
COLUMNS=120 PYENV_VERSION=research312 python -B -m pytest -q -p no:cacheprovider tests/test_script_compatibility_contracts.py tests/test_energy_budget_cli_compatibility.py
PYENV_VERSION=research312 python -B -m pytest -q -p no:cacheprovider
```

後續應先 review/提交此契約批次，再依[計畫](../refactoring/refactor_plan.md)評估下一步；
不要從通過此測試推論 signed numerical modules 可以搬移，或原始 production solver 已完全對等。
