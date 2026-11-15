# Phase 4A：Target architecture

**PROPOSED / NOT IMPLEMENTED**；source baseline 見 [index](README.md)。

## Conservative Extension topology

現有 [src/nek_post](../../../src/nek_post) scientific APIs 留原位、完整 source bytes 和 sibling lookup 不變。
現有 [pyproject.toml](../../../pyproject.toml) lines 1–13 已使用 setuptools src discovery；目前尚無新 console entrypoint、feature dependencies 或 packaged resources 宣告。

未來新增的最小責任分界（component names 是 proposal，不是既有 API）：

| Proposed component | Responsibility / boundary |
| --- | --- |
| `nek_post.cli` package with module entrypoint | argparse root/group/leaf parser, exit/status translation; console `nek-post` and `python -m nek_post.cli` share dispatch |
| static registry/parser metadata | Command names, flags, descriptions, dependency selectors and lazy handler references; stdlib-only import graph |
| configuration/resource resolver | Whole-file selection, canonical paths, separate input/output views, config identity |
| destination planner | Pure in-memory route plan, complete allowed write/delete set and native guard compatibility |
| selected workflow adapters | Join existing scientific/IO calls where script-local main currently supplies orchestration; explicit context, no protected-code movement |
| installed child runner | argv/environment/cwd/log/status/signal lifecycle for pipelines and benchmark hooks |
| invocation records | Operational execution provenance under R/.cli, separate from native scientific metadata |

不先建立 generic workflow framework、搬數值模組、替換所有 scripts，或以 unsigned helpers 繞過 protected-source identity。新 adapter 的效益與 compatibility 要逐批審閱。

## Help 與 dispatch

1. Import stdlib registry/parser metadata only；package initialization 必須維持 inert。
2. Build all root/group/leaf parsers，包含尚不可用的 discoverable routes；unimplemented command 真正執行時明確失敗，不能假成功。
3. argparse 先處理 help；help 不載入 scientific handler、YAML、configuration 或建立任何 operational files。
4. Actual operation 才 resolve scientific configuration/input identity；每個 writing invocation（含 installed dry-run）必須先由 explicit/env machine paths 建立完整 [Filesystem Safety Context](configuration_dependencies.md#mandatory-filesystem-safety-context)。Normalize/inventory → derive all destinations → safety preflight/refusal 必須先於任何 allocation、effective YAML 或 write-capable handler import；input/safety view 不受 output translation 覆蓋。
5. `--plan` 印出 in-memory command/destination/prerequisite plan 後返回，不 materialize YAML、mkdir、logs、cache、locks 或 children。其 import/bytecode/cache 行為也需 G-HELP/G-PLAN 實測。
6. Actual execute/dry-run 通過 safety preflight 後，以不匯入 write-capable handlers 的方式檢查 capability prerequisites，再分配已授權 operational paths；設定受控 cache/temp environment 後才 import selected feature handler。三個 read-only leaves 不需 writing safety context、不分配 R，仍檢查各自必要 inputs/prerequisites。
7. 根據 route 呼叫 complete API、adapter 或 installed subprocess；狀態/exception 由明確契約轉 nonzero，不把 invalid report 當成功。

完整 hierarchy 以 [49-leaf matrix](cli_migration_matrix.md) 為唯一清單，包含 `integrate gll-field`、兩個 omega validation/publication leaves、四個 results operations。

## Installed child execution

Child argv 的概念形式（ROOT_CONFIG_FLAGS 是實際 list elements，不是 shell expansion）：

```text
[sys.executable, "-B", "-m", "nek_post.cli",
 applicable effective --paths-config/--cases-config/--study-config flags,
 "--output-root", ABS_R, GROUP, LEAF, LEAF_ARGS...]
```

使用 `shell=False`；不是拼字串執行。Capture invocation cwd 後，所有 input/output/config paths 都 explicit absolute。適用 child flags 包括 input roots、case **label** 與 selection、frame indices/physical-time policy、method/threshold/sampling/chunk options、skip/overwrite controls，不能只傳 case/index 而讓 child 回到 legacy defaults。

Execution cwd 為 `R/.cli/cwd/<invocation>`；不要依賴 scripts/、checkout cwd、editable import 或外部 PYTHONPATH。通過 safety preflight 後才將 applicable effective YAML 寫在 `R/.cli/config/`；effective paths YAML 保留完整 original input/safety roots，output view 僅由 R/leaf flags 在 adapter 推導。保留 original config file hash、resolved safety/semantic identity 與 effective hash；child 使用 explicit flags，不重新讀可變 env/default configuration，並於自己 allocation/import 前重驗 safety。執行前/child 啟動前拒絕 effective identity 變更，不回填 scientific metadata。

Child environment 明確記錄與套用；包含 run-scoped Matplotlib/temp paths、no-bytecode behavior 與 native必要 locale。stdout/stderr 各自保存，記錄 argv、interpreter/version、cwd、inputs、destinations、status、exit/signal。保留科學 runtime 所需環境，不能只記錄而不套用。Environment 記錄應有明確必要欄位/敏感值處理，不無差別保存 secrets。

第一個 child failure 即停止；report 保留 failed child status，parent exit 對 signal 有明確映射，不能用 generic success 覆蓋。POSIX process-group interruption 必須向整組 child/worker 傳遞、等待/reap，必要時有界升級終止，保留 native staging/logs，不宣稱 rollback 完成。實際 timeout/escalation 時限與非 POSIX support 待 G-EXEC acceptance 決定。

Installed `--dry-run` 可分配 operational records，但不啟動 scientific children、不寫 scientific outputs；沒有 dry-run API 的 delegate 不可直接呼叫。它不同於 zero-allocation `--plan`，也不同於某些 native dry-run 會完整讀取/驗證的成本。

## Pipelines 05 / 07

已查核 checkout-bound evidence：`scripts/05_run_t19p5_pipeline.py:44–49,96–133` 與 `scripts/07_run_multitime_pipeline.py:90–150,183–195`。原 scripts 不修改。Installed parents 新增 orchestration，05 保留 extraction 後每 field compare/plot；07 保留 ordered set iteration 的 first-occurrence (case,index) extraction dedup，再按 set/field 做 compare/plot。保留實際 NPZ skip、CSV/metadata/log 計算與 overwrite 行為，不能因純 numerical API 相同就假定 script semantics 相同。

C7 先依賴 C4/C5 與相關 C6 reporting；B3 先 characterize 03/07。未來 wheel child 測試從 checkout 外執行，不能使用 legacy repository-relative scripts。

## Benchmark additive extension

`leading_edge_benchmark.py` 是 unprotected orchestration；proposed optional keyword-only extension（**目前不存在**）：

```text
compute_command_builder=None
execution_cwd=None
child_env=None
```

None 在每次呼叫時 late-select 現有 module-level builder/executor behavior，避免 default parameter 綁住 monkeypatch 前函式。None 必須保留 historical repo-script argv、cwd、環境與 subprocess.run monkeypatch path。
Installed caller 提供 module-entrypoint builder、R 下 cwd 與明確 child environment；同時接到 warmups 和 measured runs、GNU-time wrapping。Preserve LC_ALL=C、worker counts、repeats、warmups、metrics format/units、CSV schemas、artifact equality and output guards。

Source anchors：`src/nek_post/leading_edge_benchmark.py:248–321,874–904,979–1103`；tests monkeypatch subprocess.run：`tests/test_leading_edge_benchmark.py:431–500`。Case 在 `:1004` uppercased，故 **保留 case label**，不可把 absolute directory 塞進 case 欄位；child 由 explicit resolved config 找 case directory。C12 依賴 C11 installed extraction 與此 additive interface 的獨立驗收。

### Exact benchmark child output handshake (M1)

以下是 **PROPOSED / NOT IMPLEMENTED** 的 installed contract；現有 benchmark/script 19 不變。Parent 保有 authorized R；benchmark delegate 的 output directory 為 `B=R/leading-edge-benchmarks/L`。Native orchestration 在 B 下配置 `D=B/warmups/workers_W/run_NNN` 或 `D=B/runs/workers_W/run_NNN`。同一個 installed builder 將每次原生 `output_dir=D` 傳成 child 的 exact override，而 child 的 `--output-root` 始終是 parent R。

Child argv 為實際 string array（所有 paths absolute；下列大寫名稱為 resolved values，不是 shell expansion）：

```text
[sys.executable, "-B", "-m", "nek_post.cli",
 "--paths-config", ABS_EFFECTIVE_PATHS_YAML,
 "--cases-config", ABS_EFFECTIVE_CASES_YAML,
 "--output-root", ABS_AUTHORIZED_R,
 "leading-edge", "extract",
 "--case", L, "--file-prefix", PREFIX,
 "--sampling-mode", "uniform-spectral",
 "--extraction-method", EFFECTIVE_METHOD, "--x-min", EFFECTIVE_X_MIN,
 "--nx", NX, "--z-target", Z, "--threshold", THRESHOLD,
 "--y-upsample-factor", FACTOR, "--workers", W,
 "--output-dir", ABS_EXACT_D, "--overwrite",
 *INDEX_ARGS, *TIME_SELECTION_ARGS]
```

`INDEX_ARGS` 只在 bound 非 None 時加 `["--start-index", START]` / `["--end-index", END]`，保持 existing builder 的 inclusive selection；`TIME_SELECTION_ARGS` 恰為 `["--all-frames"]` 或 `["--contour-time-spacing", SPACING]`。METHOD/X_MIN 來自固定 effective cases 的 script-19 defaults，不能引入新的 benchmark method/sampling modes；現有 benchmark 是 uniform-spectral、非 explicit-ny/refined-GLL 路徑。其餘數值沿用 native builder 的 formatting/values。以 `shell=False` 執行；GNU time 的 existing format/output wrapping 同樣套用 warmup 與 measured argv。

`--output-root R` 是 authorization boundary；`--output-dir D` 是 extraction adapter 的 **exact final artifact directory**，不是另一個 root。未提供 override 才用一般 `R/leading-edge/L` 或 H selection destination；提供時不再 append case/frame/`leading-edge`。D 必須為 R 的 strict permitted descendant，通過共用 safety、symlink/ancestry、input-overlap 與 planned-product collision checks；R 外 override 一律拒絕。Parent 先 preflight B/所有 D/cleanup set，再由 native executor exclusive mkdir 每個 D 並建立 logs；child 可接收這個已由 parent 建立的 D，不能要求 D 本身不存在，也不能覆寫 parent logs、foreign files 或 immutable inputs。`--overwrite` 僅沿用 native extraction 的 run-owned artifact preflight，不取消任何安全/碰撞規則；native benchmark parent 的既有 reports/runs/warmups overwrite guards 保留。

Child 必須直接寫 `D/L_leading_edge_timeseries.csv`、`D/L_leading_edge_metadata.csv`。Reader 不改目錄、不改 filenames/CSV schemas/數值 artifact-equality algorithms。L 保持 uppercased scientific label，absolute case directory 僅由 config resolve。Warmups 與 measured repetitions 使用同一 output handshake；未來 C12 fixtures 亦須在 warmup cleanup 前讀回 CSV pair（現有 warmup source 只檢查 exit/metrics，這不是已存在的 runtime readback 宣稱）。Successful warmups 的 native cleanup 行為不變。

Source：`scripts/19_compute_leading_edge_evolution.py:122–258,277–344`；`leading_edge_benchmark.py:248–321,633–704,874–904,932–956,1054–1103,1147–1154`；`leading_edge_io.py:75–82`。C11 先驗 exact contained override/no extra nesting/outside-R refusal；C12 再驗 checkout-free wheel children、完整 argv、warmup/measured exact placement/readback、CSV/equality、None-default/monkeypatch behavior 與 protected-source freeze。詳見 [roadmap](implementation_roadmap.md)與 [G-EXEC/G-WRITE](acceptance_gates.md)。

## Scientific / runtime boundaries

CLI dispatch、destination translation、absolute input injection 不更改 scientific algorithms、operation order 或 artifact schemas。[Acceptance gates](acceptance_gates.md) 列 protected inventory 與身份限制。

Strict resume 同時看 source/config/input/output path/environment/artifact identity。Vortex source evidence：`vortex_3d_series.py:177–204,342–371,574–606`。同 bytes、wheel 可安裝、catalog completed、同 frame index 或數值看似接近，都不能替代 native eligibility。Native path guards 可能拒絕 installation ancestor 內的 output；不得 monkeypatch `__file__`、constants 或 rewrite metadata 強迫相容。

Uniform-volume VTI 與 energy closure feature-branch semantics 不是現有 dev CLI 的自動 capability；見 [ADR-0008](adr/0008-retained-branches.md)。
