# 相容性基準與 strict resume

基準與證據標記見[索引](README.md)。下列來源行號針對固定基準；完整 hash 值見 [source_fingerprints.json](source_fingerprints.json)。沒有改動任何 signature 算法，也沒有對 production 執行 resume。

## Exact source fingerprints

**CONFIRMED FROM SOURCE**：[vortex_3d_workflow.py](../../src/nek_post/vortex_3d_workflow.py) 第 41–49 行明列 18 個 basename，第 119–121 行以 `Path(__file__).parent / name` 對完整來源檔做 SHA-256。字典的 key 是 basename，value 是內容 hash；不是 package version、函式 AST、數學等價類或 Git revision。註解、空白與 import 也是被雜湊的 bytes。

```text
fields.py                    io_nek.py
gll.py                       spectral_derivatives.py
spectral_interpolation.py     gll_directional_integration.py
l2_projection.py             swirling_strength.py
rectilinear_export.py        lambda_ci_workflow.py
lambda_ci_series.py          hex_validation.py
vortex_3d.py                 vortex_3d_workflow.py
gll_refinement.py            vortex_3d_reconstruction.py
vortex_3d_artifacts.py        vortex_3d_selection.py
```

[vortex_3d_series.py](../../src/nek_post/vortex_3d_series.py) 第 84–89 行取得上列字典，再加 `vortex_3d_series.py` 與 `lambda_ci_slice_export.py`：**20 個完整 source hashes**。frame sidecar 儲存 `numerical_software_signature`；manifest/summary 儲存 `series_signature`。scripts 40/41、tests、docs 不在這兩組名單。這不是所有 transitive imports 的自動 discovery；未列入的 helper 不因此免除 runtime compatibility 風險。

| 其他已捕捉工作流程 | 精確 hash 檔案名與來源 |
| --- | --- |
| lambda_ci slices（5） | `gll.py`, `spectral_derivatives.py`, `l2_projection.py`, `swirling_strength.py`, `lambda_ci_plotting.py`；[lambda_ci_series.py](../../src/nek_post/lambda_ci_series.py) 第 123–126 行。 |
| native omega_x（6） | `gll.py`, `spectral_derivatives.py`, `l2_projection.py`, `vorticity.py`, `rectilinear_export.py`, `omega_x_workflow.py`；[omega_x_workflow.py](../../src/nek_post/omega_x_workflow.py) 第 109–113 行。 |
| saved omega_x P7（6） | `gll.py`, `gll_directional_integration.py`, `lambda_ci_plotting.py`, `lambda_ci_series.py`, `omega_x_p7_slice.py`, `omega_x_p7_series.py`；[omega_x_p7_series.py](../../src/nek_post/omega_x_p7_series.py) 第 81–85 行。 |

omega_x workflow 第 116–139 行的 exact predecessor allowlist 還核對 successor source 的 normalized digest 與其他每個檔案 hash；它不是「只要結果相同就接受」的通用版本政策。P7 的既有 legacy frame token 也只適用已列明情況，不能泛化。另 [vtu_compat.py](../../src/nek_post/vtu_compat.py) 第 504/544 行記錄及重驗自己的 source hash；五組快照不是全 repo 所有 provenance 的完整 schema。

**CONFIRMED FROM SOURCE / bounded observation**：發表的 `h_refinement/vortex_3d_series/gll10/case_N7_H/f00079/GC0_f00079_gll10.json`、`series/manifest.json`、`series/processing_summary.json` 的相應 signature 均等於本次 current 字典。三個受限讀取的實際路徑與完整 stored 值已保存在快照。這只證明 source signature 條件吻合；未重讀 81 個 VTU 或其 Nek inputs。

三個 controls 的 configuration 一致：`periodic_axes=["x","y"]`、`expected_element_counts=[272,12,8]`、`crop=[0.0,17.0,0.0,1.5,0.0,1.0]`、`chunk_size=256`、`sampling_nodes=10`；manifest 與 summary 各有 81 個 frame records。快照另保留相關 schema、source paths、selection、frame 79 的 software/Python/numerical definition（存在的欄位原樣記錄）。這是宣告的 configuration，不是獨立證明 solver 的 periodic boundary/connectivity。

## Strict resume 還綁定哪些內容

**CONFIRMED FROM SOURCE**：以 [vortex_3d_series.py](../../src/nek_post/vortex_3d_series.py) 為準：

| 層 | 實際檢查與來源位置 |
| --- | --- |
| source/header/config | `_frame_record` 第 177–205 行：case、絕對 `source_file`、index、實際 time/istep、input word bytes、element count/local shape/map、configuration、絕對 output_paths、point/cell counts、output_selection/output_plan。configuration 包括 periodic axes、element counts、chunk size、crop，sampling_nodes 存在時亦保留；series `_configuration` 與 selection 另有 series control 對照。 |
| completion/implementation | schema 1、completed、source SHA、source_unchanged、binary roundtrip、18-file signature，software dictionary、Python version、numerical_definition 完整相等。signature 相等不能替代環境相等。 |
| mesh/arrays/VTU | 第 206–257 行：native mesh coordinate order/shape/count/axis/cells/orientations、7 種 array hashes、reconstruction sampling、visualization bounds/count/types、field ranges、VTU SHA；deep path 讀回驗證 grid/arrays。 |
| namespace/identity | 第 177–183 行只允許預期 frame artifacts、拒絕 symlink；regular file identity、no-follow bounded XML fingerprint 見 [artifacts](../../src/nek_post/vortex_3d_artifacts.py) 第 22–97 行。不是只檢查檔案存在。 |
| manifest/checkpoint | `_control_files` 第 326–339 行；`_preflight` 第 342–473 行比對 workflow/schema、source_case、configuration、selection、20-file signature；manifest frame index/time/source/meta/VTU hashes 及 mesh 必須一致。 |
| PVD/time | `_pvd_entries`（第 260 行）、`_publish_pvd` 與 preflight 檢查解析出的 frame 路徑與實際物理時間；既有 control set 不能有 foreign entries，不能用 index 代替 time。 |
| stages/locks/publication | `run_vortex_series` 第 574 行起先 read-only preflight，第 597 行在 persistent PVD lock 下重新 deep preflight、重驗來源/產物/signature，再做一個 frame；完整 staging 驗證後才可 recovery。新目錄使用 [lambda_ci_series.py](../../src/nek_post/lambda_ci_series.py) 第 344 行 `renameat2(RENAME_NOREPLACE)`；PVD/JSON 暫存 fsync 後替換有各自鎖與一致性規則。 |

**SUPPORTED BY TESTS**：[series tests](../../tests/test_vortex_3d_series.py) 包含 `test_changed_source_or_series_signature_stops_resume`（第 301 行）、`test_strict_resume_preserves_existing_artifacts_and_reconciles_nonconsecutive_pvd`（149）、staging/crash/lock tests（369 起、488 起）；[artifact integrity](../../tests/test_vortex_3d_artifact_integrity.py) 檢查 publish 前後 namespace substitution。移到別處並重寫 sidecar 可以是另一項已稽核資料遷移，但本計畫不使用這種方式保住 source-refactor resume。

## A–H 操作影響矩陣

「fingerprint-safe」只描述指定 signature 的 source 字典，不表示整體行為已證安全；尚未執行任何未來 refactor。

| 操作 | 已確認機制與結論 | 證據強度 / 未來接受條件 |
| --- | --- | --- |
| A. 原封不動搬 source | 同 bytes 的純 hash 相同，但查找是 workflow `__file__` 旁的 basename。搬走個別 signed file 會缺檔；連 workflow 同搬仍有 `parents[2]` repository depth、imports、config、qualname 風險。 | **CONFIRMED FROM SOURCE**：hash bytes 規則；整體 relocation **UNVERIFIED**。不批准搬 signed files。 |
| B. 修改 import | 若檔案在某組 signature 中，其 hash 必變，對應歷史 exact signature 不再相符；未簽名 consumer 的 import 改動不直接改那組字典，但可能換實作。 | signed case **CONFIRMED FROM SOURCE**；behavior **UNVERIFIED**，要 synthetic regression。 |
| C. compatibility wrapper | 用 wrapper 取代 signed source，哪怕 re-export 同函式，仍改 bytes/hash。wrapper 的 `__file__`、globals、monkeypatch、pickle identity 也不等同原 module。僅新增 unsigned CLI wrapper 不直接改上述字典。 | 前者已知 invalidation；後者只屬 signature-safe inference，需 help/import/CLI/output tests。不可對 signed core 用 wrapper。 |
| D. 搬 CLI | 40/41 不在 vortex source list，純 script relocation 不直接改這 20 hashes；其 root-depth、subprocess caller、test/import/docs/historical command 仍會壞。 | 字典 membership **CONFIRMED FROM SOURCE**；整體安全 **UNVERIFIED**。保留舊 script path。 |
| E. module-qualified 名稱改變 | signature 不另存一般 callable qualname；通常 rename/import 修改 signed source 已造成 invalidation。worker pickle/spawn 或外部 serialized imports 另會受影響。 | source schema 已確認；外部 consumers **UNVERIFIED**。保持原 import 與類別/worker qualname。 |
| F. 搬 tests/docs | 不直接改上述 signatures，但 file-based CLI import、sibling test imports、reference driver、repo-cwd evidence 路徑、Markdown 相對 target 可壞。 | membership 已確認；目前 tests 運作 **SUPPORTED BY TESTS**；搬後安全尚未證明。先導覽，不先搬。 |
| G. 新 helper | 單純增加未被引用的檔不改明列字典。由 signed module 匯入它會因 import bytes 改變而 invalidation；helper 不會自動加入 coverage。由 unsigned caller 使用也可能改科學行為。 | **CONFIRMED FROM SOURCE**：固定清單；新 behavior **UNVERIFIED**。不得拿未簽名 helper 迴避 signature。 |
| H. 數學相等的重構 | 只要 signed file bytes 改變，exact signature 不同；通過數值 tolerances 不會恢復歷史 source identity。 | 已知 fingerprint invalidation，**CONFIRMED FROM SOURCE**；延後，不能放寬 compare 或回填 metadata。 |

## 介面契約與必要回歸

| 現有介面 | 固定契約 / 測試 | 未來結構與 rollback |
| --- | --- | --- |
| 46 舊 script paths、argparse | 完整 help、flags/choices/default/type、錯誤 exit/stderr、repo/config resolution；snapshot 僅為本環境 baseline，路徑不能隨意正規化後視為同等。05/07 與 benchmark subprocess caller、動態 import 測試需保留。 | 舊路徑留存；只小批抽取 unsigned CLI，見[計畫](refactor_plan.md)。失敗時還原未發表批次或以已批准 revert 復原該未來 commit，不碰科學資料。 |
| `nek_post.*` public 與實際跨模組 underscore imports | inventory definitions/signature/import consumers；代表：`gll_nodes(node_count)`、`physical_gradient(values, coordinates, *, singular_rtol=...)`、`project_scalar_field(operator, values, *, mesh=None, out=None, chunk_size=...)`、`project_velocity_gradient(..., diagnostics=True, ...)`、`compute_swirling_strength(velocity_gradient, *, chunk_size=...)`。精確 annotation/default expression 以 inventory 與 source 為準。 | signed modules 位置/內容皆留；unsigned reporting extraction 保留舊 import export、signature、globals/monkeypatch 語意，否則取消 extraction。 |
| lambda/omega/P7 科學定義 | 保留 Phase 1 analytic references。curl 後 scalar recovery 與 gradient tensor recovery 分開；lambda_ci 在 projected tensor（dense variant 先重建 tensor）上求 eigenvalues，不投影 eigenvalues。 | 不更動 numerical methods。幾何/BC caller responsibility 見[方法](../methods/vortex_numerical_method.md)，不新增 topology rejection 承諾。 |
| writer/schema | native VTR、element-local binary VTU、P7 reconstructed arrays、time PVD；dtype/order/shared-point policy、positive hex/Jacobian、roundtrip、bounded resources、no-overwrite。 | 保留 existing writers/schema/native/P7 的數學/serialization 分界；用 synthetic fixture 比較 arrays/schema，不能重算 production 來更新 golden。 |
| catalog schema/identity | [results_catalog.py](../../src/nek_post/results_catalog.py) 第 12–30 行：schema 1，25 個 DATASET_COLUMNS（順序固定），7 個 ARTIFACT_COLUMNS。第 241 行 dataset_id = `ds_` + SHA-256(relative POSIX dataset path) 前 20 hex；搬 dataset 會改 identity。 | catalog/publication 不搬不改；nested-run boundaries、manifest association、malformed/missing reference、deep JSON regressions 必須保留。 |
| publication | [publication.py](../../src/nek_post/results_catalog_publication.py) 第 69–100 行：只獨占建立 README/CSV；與 fresh catalog 完整比較，pair 非 atomic transaction，失敗可留下 partial 檔並阻止 retry。 | 不順便「統一」為 vortex publishing；本階段 production root readonly。 |
| strict resume | 上述 source/config/environment/path/source+artifact hashes、locks/staging/PVD；不能以 test pass 或 catalog completed 代替。 | 同一 baseline source 維持 bytes/import layout；所有未來 batch 比對五組 map，並跑 synthetic resume rejection/preservation tests。 |
| docs/tests paths | README/script index/methods/results guides/local links、sibling fixture imports、driver/evidence relative paths。 | 新增索引優先；任何搬移需逐檔 caller/link 修復，保留歷史舊入口；rollback 僅該批 repo changes。 |

## 81-frame collection 保護策略

1. **CONFIRMED FROM SOURCE**：保留目前 20 個 vortex signed 檔及其 sibling 路徑、byte contents；其他五組相關 signed 檔亦凍結。Phase 3B/C/D 只選不在這些集合中的導航/unsigned reporting 工作。
2. **INFERRED**：若日後必須重構 signed code，將既有 baseline commit 與環境保留為唯讀可重建執行版本（例如另建獨立 checkout/environment，需另行授權）；新研究使用獨立版本與新 output root。維持舊 metadata 原樣；新版本不能冒稱可以 resume 舊 collection。
3. 不改 stored signatures、不放寬 eligibility、不修改科學 sidecar、PVD、manifest、checkpoint 或 source path。現有 collection 的閱讀/catalog 不要求 rerun producer。
4. future synthetic tests 必須驗證 source/import/wrapper hash 變動被拒絕、舊 artifact bytes 保留、no-recompute resume、lock/staging failure；任何想要歷史相容的設計需獨立 review。本次只完成目前 signature 比對，沒有宣稱 production resume 已全數通過。
