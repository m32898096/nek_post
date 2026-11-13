# Phase 3B / 3C / 3D 分批方案

本檔所有新路徑均為 **INFERRED** 的未實作提案。Phase 3A 沒有建立 wrapper、抽取函式或搬動檔案。批准下一階段前先閱讀[相容性矩陣](compatibility_matrix.md)；每批獨立 review，不能一次重寫儲存庫。

## 目標結構與不搬清單

```text
scripts/                     # 所有現有 numbered/unnumbered 路徑保留
src/nek_post/                # 現有 import API 與 signed source 保留
  cli/energy_budget.py       # 3C 小型試點，尚不存在
  reporting/energy_budget.py # 3D terminal text formatter 試點，尚不存在
tests/                       # 先保留 flat collection/sibling helpers
  numerics/                  # 3D 選擇性 energy unit test 試點，須先證明可攜
  reference/                 # 現有 Fortran driver 保留
docs/methods/                # 科學定義與 provenance，保留
docs/results_catalog/        # 產品導覽與詮釋，保留
docs/guides/README.md         # 3B 新導覽，不搬舊 workflow 文件
docs/refactoring/            # 本稽核與後續決策
```

下列目前有清楚責任或高 provenance 風險，建議**保持現況**：`gll.py`、`spectral_derivatives.py`、`l2_projection.py`、`vorticity.py`、`swirling_strength.py`、`gll_refinement.py`、spectral interpolation/積分、所有 vortex/omega/lambda workflow/writer、`hex_validation.py`、`vtu_compat.py`、catalog/publication。精確 signed 清單見矩陣；不能只按檔案主題決定可否修改。

## Phase 3B：先改善導覽與相容性檢查

| 批次 | 原始 → 提議位置/內容 | 原入口與相容層 | 驗證 / hash / 風險 / rollback |
| --- | --- | --- | --- |
| 3B.1 | `README.md`、`docs/script_index.md` 保留原位；新增 `docs/guides/README.md`，依任務連結現有 guides。README 只更正 minimal-tests/proposed-validation 導覽文字，不改科學結論。 | 既有連結/歷史命令不改寫；不用 redirect 或 wrapper。責任是導航，非重新敘述方法。 | 檢查所有新增/更動 local links 的 target 語意、Phase 1/2 claims，跑 catalog guide tests、whitespace/full suite；五組 hashes 完全相同。低風險；rollback 只還原這批導航文件。 |
| 3B.2 | 目前隱含的 CLI/import/resume 契約 → 新 `tests/test_repository_compatibility.py`（提議），以本次 46-script/5-signature inventory 決定小型穩定 assertions。 | 不修改現有 tests/reference helpers，不建立 `tests/__init__.py`。不要把本機 help 的絕對路徑硬編為跨機 golden；固定 flags/default semantics，config 由 temp fixture 提供。 | 不觸碰 scientific root；source signature equality/rejection 用 isolated monkeypatch/fixture，不改真 source。測試失敗停止；不更新 golden 掩蓋差異。低至中風險（brittle snapshots）；無有效獨立契約就不新增測試。rollback 移除該未來新增測試 commit 的變更。 |

3B 完成條件：舊路徑皆存在、沒有 source/CLI 改動、既有 2,786 baseline cases 不消失、所有 signature maps 不變、navigation review 通過。新增測試數獨立記錄，不能只比總數相同。

## Phase 3C：一個 unsigned CLI extraction 試點

只提議先處理 **script 12**，不用同時重整 scripts 00–44。原因：它的能源數值、IO、plot 已分離，main 只組織這些呼叫；檔案不在五組 captured source signatures。收益是可直接測 CLI orchestration；若單一額外 module/wrapper 的維護成本大於收益，可取消，不擴展批次。

| 批次 | 原始 → 提議位置/抽取 | 舊介面是否保留 / 相容要求 | 必須先證明的內容 |
| --- | --- | --- | --- |
| 3C.1 | `scripts/12_check_energy_budget_closure.py` 的 orchestration → `src/nek_post/cli/energy_budget.py`；新增空 package `src/nek_post/cli/__init__.py`。數值/IO/plot 不動。 | 舊 script **必須留**。保留 `main`、`_parse_args`、`_parse_cases` 的呼叫契約及實際 import 使用的符號；明確傳原 repo/config context，不能把新 module 的 `__file__` depth 當舊 script root。不要新增 console entrypoint 取代舊命令。 | 先用 file-based import 調查實際 globals/monkeypatch；普通 re-export 不會自動維持 wrapper globals patch 對 implementation globals 的效果，無法簡單保留就取消 extraction。新 API 不自動等同 external private ABI。 |
| 3C.2 | script 12 能源 CLI 的隱含行為 → 新 `tests/test_energy_budget_cli.py`（提議）。舊 `tests/test_energy_budget*.py` 不搬。 | fixture 建立小型 DAT/temp outputs，透過舊命令及新 implementation 比較；外部腳本路徑照舊。 | `--help` flags/default/help、`--cases` parsing、`--target`、`--no-plots`、output-dir/overwrite、失敗 exit/stderr、CSV header/precision/row order、figure filenames、terminal table 保持相同。另跑現有能源/plots/paths 與完整 suite。 |

**Fingerprint impact**：不改任何已捕捉 signed source、map 須 byte-for-byte 相等。這只直接證明 hash 不變，不能取代數值/CLI 行為驗證。script path 可能出現在外部歷史 provenance，故保留舊檔名；不回填 producer_script。風險中等（config depth、global identity、argparse help `prog`/exit）。新 parser 要保留舊 script 的程式名稱顯示，不能無聲改成 module 名稱。

**Rollback**：future batch 若失敗，停止擴展；在尚未提交時只修正/還原該批已確認變更；已發布則經授權新增 revert commit。不要 reset、stash/clean 或修改 scientific outputs。恢復 script 12 舊實作；移除新 module 的 caller 依賴，不撤掉其他批次已接受的 tests/導航。

## Phase 3D：有限 reporting 分界與測試試點

3C 通過獨立 review 後才開始；這不是核准拆分大型數值/cache 模組。

| 批次 | 原始 → 提議位置 | 保留介面、理由及風險 | 回歸 / provenance / rollback |
| --- | --- | --- | --- |
| 3D.1 | `src/nek_post/energy_budget_io.py::format_energy_summary_table` → `src/nek_post/reporting/energy_budget.py`；新增 `reporting/__init__.py`。`format_numeric_value`、CSV writers 與 constants 先留原處。 | IO 中純 terminal-text formatting 可獨立。舊 `nek_post.energy_budget_io.format_energy_summary_table` 保留同 signature 的 facade；如果新模組反向依賴 IO，必須 lazy dispatch 或改為明確參數傳入，避免 eager cycle；不能為搬 1 個函式新增大片抽象。現有 monkeypatch/formatter globals 語意先測，否則保持現況。中風險；效益未證前可不做。 | 現有 `test_energy_budget_io.py` 第 133–208 行固定 CSV/16-digit formatting/table title/order；另加 facade/import/no-new-cycle tests。CSV與數值函式不變；五組 hashes 相等。rollback 僅將此 formatter 放回舊 module 並取消新依賴。 |
| 3D.2（可選） | `tests/test_energy_budget.py` → `tests/numerics/test_energy_budget.py`；其他 energy IO/plot tests 先留。 | 選 pure energy kernel test 作為低 path-coupling collection 試點，不能兩處重複 collect。已檢查此檔沒有 `__file__`、dynamic script 或 sibling test imports；外部指定舊 pytest 路徑仍可能存在，屬 **UNVERIFIED**，不確定就延後搬。原測試檔路徑不提供假 wrapper，避免重複收集。 | 記錄逐 case node IDs 的一對一對應、單檔與全 suite collection/結果、repo cwd 及任意 cwd 的已支持情境、無 helper import break。source fingerprints 不變；中風險（外部 test commands）。rollback 只還原該檔位置，不改 assertions/goldens。 |

若 3D.2 的分類無法明顯提升可讀性，選擇只做邏輯 test index 而不搬檔。其餘 proposed test groups 先留為導覽分類，不把 test movement 當成完成指標。

## 必須延後或避免

- **已知會破壞 exact fingerprint**：改寫/搬出 signed source、signed file wrapper、跨 signed module 的 import cleanup、數學相等的 kernel extraction。未提出保留 legacy runtime 與新 output-root 的獨立方案前不做。
- **未證安全**：一次搬全部 numbered scripts、切換 public import hierarchy、移除 underscore helpers、把 lazy cycle 改成 eager imports、換 pytest import mode、共用 fixture 全面搬移、改 worker class/function qualname。
- **收益未證**：為 atomic JSON、rename、frame parser、CSV 或 plot 相似性引入 generic framework；不同 workflow 的 lock/schema/partial failure/overwrite 規則不可消失。
- **不在重構範圍**：dataset relocation/renaming、source/metadata fingerprint rebinding、golden regeneration、catalog schema/identity 改版、再產生 81-frame series、production solver-equivalence 宣稱。

## 各批共同接受門檻

1. 開始前 branch/HEAD/status 與批准範圍一致；保存五組 signature map 與逐檔 diff。任何 unrelated change 停止，不自動復原使用者工作。
2. 比較舊 CLI help/argparse、public/實際 private imports、代表 signatures、globals/worker identity；適用時對比 isolated output bytes/schema，不讀 production payload。
3. 跑相關 unit/CLI/serialization/resume/numerical/catalog tests，再跑 `PYENV_VERSION=research312 python -B -m pytest -q -p no:cacheprovider`，temp artifacts 在 repo 外。保留 independent analytic expectations，不用新實作反算參考值。
4. Markdown target、syntax/import、whitespace、逐檔 scope/source hashes 通過；source signatures 預期不變的批次若 map 不同，立即停止並調查，不能修改 stored signature 消除差異。
5. scientific root 全程 readonly；production metadata/PVD/LOCK/checkpoint 一律不碰。不以 production resume 作為重構 smoke test。
6. 每批列出實際相容性證據、未驗證 external caller、rollback 範圍，獨立 review 後才考慮下批；此文件本身不授權 commit/push/merge 或 Phase 3B。
