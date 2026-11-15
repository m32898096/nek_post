# Phase 4A：Configuration、resources 與 dependencies

**PROPOSED / NOT IMPLEMENTED**。既有 legacy configuration behavior 不變。現有 `config.py:11–27` 只讀 YAML/組合 mapping；`paths.py:38–95` 不實作以下 resolver。

## Per-file precedence 與 identity

| File role | Explicit flag | Environment | Fallback |
| --- | --- | --- | --- |
| machine paths | `--paths-config` | `NEK_POST_PATHS_CONFIG` | Actionable error；paths.example.yaml 只供參考 |
| scientific cases | `--cases-config` | `NEK_POST_CASES_CONFIG` | Packaged cases.yaml |
| scientific H study | `--study-config` | `NEK_POST_STUDY_CONFIG` | Packaged h_refinement.yaml |

Scientific configuration 只 resolve actual operation 所需 files；**每個 writing invocation（含 installed dry-run）另必須 resolve machine paths 作為 Filesystem Safety Context**，即使 scientific delegate 只用 explicit paths。來源是 `--paths-config`，其次 `NEK_POST_PATHS_CONFIG`；缺少來源即拒絕，沒有 installed machine-specific fallback。Whole-file override，不先 merge defaults/逐 field 繼承；explicit/env 指向不存在、無效 schema、empty/錯型別或無權讀取的檔案必須失敗，不靜默 fallback。

Capture invocation cwd 一次；relative explicit/env configuration filenames 以該 cwd 解譯。Config filename resolve 到 canonical file 後，其 **canonical parent** 是相對 filesystem-valued YAML entries 的 anchor：若 symlink 位於 A 指到 B/config.yaml，值跟隨 B，不跟 A。直接 CLI input/output relative paths 跟 invocation cwd，不跟 config directory。保存 supplied filename、canonical path、原檔 hash 與 resolved mapping identity以解釋此差異。

## Current field-specific path schema

以下是 current [ProjectPaths](../../../src/nek_post/paths.py) lines 38–95 與 [paths.yaml](../../../config/paths.yaml) lines 1–19 的全部 filesystem-valued 欄位。
`paths.` 是 `load_project_config` 組合後的 logical prefix；原 paths.yaml 的 top-level 是 data_root/case_dirs 等，不額外包一層 paths。

| Logical field | Normalization |
| --- | --- |
| `paths.data_root` | One directory |
| `paths.case_dirs.<case-label>` values | Each directory value；key 是 scientific label，不改寫 |
| `paths.postproc_root` | One output/input storage directory by view |
| `paths.results_root` | One results directory by view |
| `paths.h_refinement_results_root` | Explicit absolute H root in output view |
| `paths.paper_data.cantero_fig5a_re3450_csv` | One input CSV |
| `paths.paper_data.cantero_fig5a_re8950_csv` | One input CSV |

Scientific cases.yaml 與 h_refinement.yaml 的現有 labels、methods、indices、Reynolds numbers、thresholds、tolerances、grid counts 不是 paths，不作字串泛化 resolve。未知 schema extension 必須明確評估，不把所有 YAML strings 當 paths。現有 H-root fallback 由 data_root/results/h_refinement 推導（`paths.py:58–62`）；新 output view 明確填值，避免外寫 machine data root。

## Mandatory Filesystem Safety Context

此為 **PROPOSED / NOT IMPLEMENTED** 的所有 writing routes 共用前置條件，不取代 scientific configuration，也不因 delegate 不讀 machine paths 而豁免。使用上述 explicit/env machine paths 及 invocation 的 explicit inputs 建立下列帶角色的完整 inventory；使用原始 input view，不能先套 output translation。

| Inventory source | Safety role / boundary |
| --- | --- |
| `data_root` | Dataset container/ancestry anchor；不得以其本身或其祖先作 R、刪除或覆寫該 container。它不是所有 descendants 一律唯讀的宣告；明確授權且與下列 protected trees 分離的 descendant R 仍須通過 native guards |
| Every `case_dirs` value | Raw immutable case trees，包含未選案例；不得把 inventory 縮成 selected case。Mapping 必須非空、entries 完整且型別/路徑有效，適用 scientific cases/study 所需 label 缺失即拒絕；無法建立完整 required inventory 時 route unavailable |
| Original `postproc_root`, `results_root`, `h_refinement_results_root` | 保護既有 input/historical artifact trees；R 與其不得重疊。H 欄位可依既有 schema 省略，但必須先按 `paths.py:58–62,77–89` 從 normalized `data_root/results/h_refinement` 算入 inventory；不是省略保護 |
| `paper_data` CSV values、explicit CLI input files/directories、其他 declared references/configuration | Immutable input/reference roles，按實際 file 或 tree 保護；不把 CSV 的所有兄弟目錄一律當 immutable。保護有效配置的 identity；不得寫回 inputs 或以 output view 取代 input lookup |
| Source checkout、installed package/module locations、native protected installation ancestors | 不接受 scientific/operational output；保留 native `__file__` ancestry rules，不以換 cwd 或 monkeypatch 繞過 |
| Explicit R；publish 時額外 explicit P、reviewed V | R 只授權完整 plan 中的 scientific/operational writes；P 只例外授權兩個 navigation targets，V 在 publish 唯讀。P 不豁免 raw/reference/package 保護，也不授權其他 historical writes |

新 output view 是另外由 R 推導的授權角色，不將原始 historical roots 改標為可寫。Run-owned intermediate/config files 可位於 R，後續 child 將它們列為 immutable inputs；允許共用 R 不等於允許 write/delete set 與該 inputs 相交。Resume/control updates 僅限該 run 的 native protocol，不能藉此修改 historical inputs。Inventory completeness 以顯式配置及適用 selections 為邊界，不宣稱自動發現未配置的外部 datasets。

Writing preflight 固定順序：lightweight parse → resolve mandatory safety source → normalize/validate recognized fields → build complete protected inventory → derive R、route products、operational paths、P/V 與 exact overrides → check overlap/ancestry/traversal/symlinks/authorization/native compatibility → refuse incomplete/unsafe/Unknown plan → check prerequisites without write-capable feature imports → 才可 allocation/effective-YAML materialization、設定受控 cache/temp environment、import feature handlers 與 processing。失敗不得建立 R/.cli、logs、caches、locks、staging 或 temporaries；child 亦須在自己分配/匯入前重驗固定 safety identity。

Help 不讀 configuration、不做 allocation。Writing route 的純 `--plan` 會讀安全配置並在 memory 檢查/報告計畫，缺失則報錯，仍不建立目錄或執行 scientific workflows。三個 read-only leaves（`h-refinement inventory`、`omega-x validate-series`、`results inventory`）不需 writing safety context，也不得建立 operational files：前者仍需 scientific P,H；omega 可用 explicit absolute source/input paths 而不需 machine config；results 只需 explicit scanned root。任何新增 report/log/cache allocation 都使 route 成為 writing，必須先滿足共用安全契約。

逐 leaf safety requirements 見 [CLI matrix](cli_migration_matrix.md)；B2 與 G-WRITE 的 mandatory failure/zero-allocation cases 見 [roadmap](implementation_roadmap.md)及 [acceptance gates](acceptance_gates.md)。

## Packaged resources 與有效配置

Proposed `nek_post.resources/cases.yaml`、`h_refinement.yaml`、`paths.example.yaml` 以 importlib.resources logical Traversable API 讀取，不依賴 cwd 或假設 resource 永遠有長期 filesystem path。必要 temporary extraction 的生命週期要有界；plan/help 不做 resource filesystem materialization。cases/study 保留 source scientific content；example 不自動啟用 machine-specific /data settings。

Setuptools package-data 必須在 wheel 和 sdist 都含 YAML；metadata/distribution 檢查與 isolated-wheel test 為 G-CONFIG，不於本次 build。Current pyproject.toml 尚未具備這些新增設定。

Input/output 兩個 immutable resolved views：input view 保留 datasets、paper CSV 與 explicit saved-artifact roots；output view 指到 R 下 [route destinations](destination_contracts.md)。不得把 input results roots 全域改成新 output root 導致誤讀新空目錄。API 支援位置傳 absolute case_dir/input_dir/output_dir；不支援位置使用 explicit workflow adapter，不能改 protected code。

Effective YAML 使用 native raw-file schema；child 的 effective paths YAML 保留完整 normalized original machine/input/safety inventory（含 derived H input root），不將 historical postproc/results roots 改成 R 的 output view。Output translation 由 explicit R/leaf destinations 在 adapter memory view 與 invocation record 表達；run-owned intermediate inputs 由 explicit input flags 傳入。重讀後 case labels、indices、thresholds、methods、orders、tolerances、排序與 typed values 必須等價，不能只比較 YAML text hash。materialization 僅在 writing execute/dry-run 通過 safety preflight 後的 R/.cli/config；plan 只在 memory。原檔 hash、完整 safety identity 和 effective identity 同時記錄，changed identity 拒絕，不重綁 native manifests。

Protected case resolvers 支援 absolute paths：`omega_x_workflow.py:96–100`、`vortex_3d_workflow.py:124–128`。它們保留 legacy alias/defaults，CLI 不靠這些 machine defaults。Benchmark 是例外：保留 uppercase case **label**，在 installed child 透過 explicit config resolve，而非傳 absolute case 到 uppercase 欄位。

## Dependency profiles

| Profile key | Proposed requirements / qualification |
| --- | --- |
| core | NumPy, SciPy, PyYAML, pymech；不是宣稱每個 metadata-only leaf 都用到全部 |
| plot | Core plus Matplotlib/Pillow |
| vtk | Core plus VTK；apply only to selected export/readback capability |
| volume | Core plus VTK/PyVista and transitive dependencies；也用於 VTU diagnostics |
| poppler | External pdftoppm；驗證 PDF completion 的路徑必須先檢查 |
| time | GNU time，native default /usr/bin/time，不能以其他同名工具假定 format 相容 |
| development/reference | pytest、packaging/build tools、optional Fortran reference prerequisites；不混入 normal runtime |
| native OS/filesystem | POSIX fcntl/resource；Linux libc renameat2(RENAME_NOREPLACE)、hard links、fsync、native locks/filesystem semantics 按 route 檢查 |

Per-leaf required profile keys 在 [migration matrix](cli_migration_matrix.md) 最後一欄逐一列出，49 leaves 都有條目。所有 leaf 共享 G-HELP/G-CONFIG（若適用）；實際 command execution 才查 prerequisites。No-plots 不自動取消 eager dependency；逐 branch characterize 後才可縮減需求。

Source-specific evidence：

- pymech lazy imports：`io_nek.py:23,30`；NumPy/SciPy import timing 依 selected module，不在 root help 載入。
- H report/analysis IO eagerly selects Agg/imports pyplot：`h_refinement_comparison_io.py:9–15`、`h_refinement_analysis_io.py:8–10`。
- Lambda series Pillow/Poppler：`lambda_ci_series.py:144–159,241–247`，slice exports/readback need VTK。
- P7 **single frame** calls PDF validator at `omega_x_p7_slice.py:334–336`；series 亦驗證圖片/PDF，不是 optional production completion dependency。
- VTU diagnostics imports PyVista/VTK in selected functions：`vtu_compat.py:146,167–168,182,291`；vortex read/write likewise `vortex_3d.py:121,173–174`。
- Native directory publisher requires Linux renameat2：`lambda_ci_series.py:344–354`；locking at `:334–340`。Supported platform matrix 的最後範圍待 B3/4D 實測，不能 claim cross-platform production support。

## Help/backend/environment contract

Static stdlib registry先讓 argparse 處理所有 help；不 import handlers、resolve config、建立 R/.cli 或 caches。Unavailable leaves remain discoverable，execute 明確報未實作。

Existing plot modules可能 eagerly set Agg、setdefault MPLCONFIGDIR 或 import pyplot；例如 `front_detection_plotting.py:9–15`。**不承諾保留 in-process caller backend**。Standalone CLI subprocess 隔離 process-global state；programmatic calls可能影響 backend。

Writing headless invocation 在 import 前明確設定 MPLCONFIGDIR 到 R/.cli/matplotlib；不能只用 setdefault，因 inherited external value 可能外寫。適用 temp env 指到 R/.cli/tmp，但 native same-directory staging 不搬動。Externally launched tools/VTK/PyVista 的完整 cache/write behavior仍須 G-WRITE 觀測；未知外寫 fail closed。

安裝成功與 dependency import 成功不證明 historical strict resume eligible。Native software/Python/config/path/source/artifact equality checks 原樣保留，與 feature availability 分別報告；`vortex_3d_series.py:187–204` 明確比對兩者。
