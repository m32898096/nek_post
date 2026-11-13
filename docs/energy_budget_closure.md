# Energy Budget Closure

## 入口與相容性

固定入口是 [scripts/12_check_energy_budget_closure.py](../scripts/12_check_energy_budget_closure.py)，
從儲存庫根目錄執行；先依根 README 完成 editable install。它讀取根目錄
`config/paths.yaml` 的預設路徑，CLI 協調以下既有模組：

| 模組 | 責任 |
| --- | --- |
| [nek_post.energy_budget](../src/nek_post/energy_budget.py) | 能源收支數值計算與 summary 資料結構 |
| [nek_post.energy_budget_io](../src/nek_post/energy_budget_io.py) | DAT 讀取/驗證、輸出命名、CSV 與終端表格 |
| [nek_post.energy_budget_plotting](../src/nek_post/energy_budget_plotting.py) | 既有診斷圖 |

Phase 3B Batch 1 只整理導覽，script 與上述模組均未搬移或修改。
`--cases` 預設 `N5,N7,N9`、`--filename` 預設 `energy_budget.dat`、`--target` 預設 `12.0`；
`--data-root` 與 `--output-dir` 可覆寫設定路徑，`--no-plots` 可略過繪圖。
既有輸出需要明確 `--overwrite`；本批保留 CLI、可匯入符號及錯誤行為。
下方既有 production 範例保留供查閱；相容性測試僅使用 pytest 暫存 DAT/CSV，
不執行這些 production 範例。

[CLI 相容性測試](../tests/test_energy_budget_cli_compatibility.py) 保護 help、預設參數、
可匯入入口、synthetic 輸出與缺檔/覆寫拒絕；既有
[數值測試](../tests/test_energy_budget.py)、[IO 測試](../tests/test_energy_budget_io.py) 與
[繪圖測試](../tests/test_energy_budget_plotting.py) 持續保護原方法與格式。
回到[任務導覽](guides/README.md)或[腳本索引](script_index.md)。

The teacher-provided `energy_budget.dat` files contain five columns:

```text
time E_k E_p E_total epsilon
```

Here `epsilon` is treated as the dissipation rate. The main closure check is:

```text
E_total(t) + integral epsilon dt ~= constant
```

For the current Re3450 cases, the expected constant is approximately `12`. The script also computes a pointwise differential check:

```text
dE_total/dt + epsilon ~= 0
```

If `energy_closure = E_total + integral epsilon dt` is nearly constant but offset from `12`, the budget is internally consistent but may use a different normalization. If `energy_closure` drifts strongly, there may be an energy budget mismatch, sign issue, or integration issue.

Run the check with:

```bash
python scripts/12_check_energy_budget_closure.py \
  --cases N5,N7,N9 \
  --target 12 \
  --overwrite
```

By default, inputs are read from:

```text
/data/Nek5000_data/case_N5/energy_budget.dat
/data/Nek5000_data/case_N7/energy_budget.dat
/data/Nek5000_data/case_N9/energy_budget.dat
```

Outputs are written under:

```text
/data/Nek5000_data/results/poly_order_compare/energy_budget_closure
```

## Teacher-focused N7 figures

Generate the requested N7-only figures with:

```bash
PYENV_VERSION=research312 python scripts/12_check_energy_budget_closure.py \
  --cases N7 \
  --target 12 \
  --overwrite
```

`energy_budget_components_N7.png` shows `E_k`, `E_p`, `E_total`, `epsilon`,
`energy_closure`, and `target 12` on one shared y axis spanning `0–15`.

`energy_closure_vs_time.png` uses the y range `10–13`. When the command is run
with `--cases N7`, its only case curve is N7, alongside the `target 12` line.
