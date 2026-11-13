# Phase 3C Batch 1：Directional GLL 測試輸入的歸屬

基準為 `ebc7e31a7d026ec2afe56bcb1b53a9b3b758cb08`。
原 `test_integrate_gll_field_cli.py` 為使用 `_data`，必須匯入含測試案例的
`test_gll_directional_workflow.py`。兩者現在直接從
[gll_directional_test_support.py](../../tests/gll_directional_test_support.py)
匯入 `make_directional_data`，並保留各自的 `_data` 名稱。

builder 的參數、函式內容及每次呼叫建立新陣列的行為保留；包含原有 GLL 節點、
`z,y,x` 順序、錯誤 concentration shape 與指定速度分量。獨立解析期望值、assertions、
容差及 artifact schema 不變；helper 只建立測試輸入，不計算參考答案。
它是普通 test-only module，不是 pytest fixture，也不由 pytest 當成測試檔收集。
未建立全域 conftest、tests package 或修改 pytest import mode；import 時不寫檔。

本批只解除這兩個測試模組的單向依賴。其他 sibling-test imports、外部對 private
test helpers 的依賴與未來搬檔後的 import 行為仍須個別評估。production source、
numbered entrypoints、fingerprints、strict resume 與科學產物均不在變更範圍。
