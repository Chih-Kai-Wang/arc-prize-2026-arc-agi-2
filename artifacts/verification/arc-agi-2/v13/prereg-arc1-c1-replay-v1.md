# 預登記：ARC-1＋C1 零 GPU 重播（2026-10-19 07:0xZ，寫在任何程式與任何 keys30 結果之前）

來源：協調者 10-19 新方向指示（使用者解除暫停）；Argon `artifacts/plans/argon/newdir-arc-agi-2-20261019/argon-r1.md` §C0／C1；advisor 07:2xZ。

## 0. 先記下的發現（限制）
- E3／E6／v2／v3／v4 的 `v13-programs.jsonl` 只存 `code_tail = code[-1500:]`（`v13_llm.py:488`）；49 筆驗證過的程式**沒有一筆有完整程式碼**。
- → C1 第 1 部分（把程式套在 D4／換色後的測試輸入上檢查等變性）**無法離線重播**，「真陽性程式被誤殺」也量不到。它最多只能在 kernel 執行時當 CPU 檢查；**對下方 8.253 閘門的貢獻記為 0**。
- 下一個 kernel 一定要另存完整程式（例如 `v13-code.jsonl`）。

## 1. 元件（都只用 CPU）
- **R（C1 第 2 部分）NVARC 候選池的不變量過濾**：從 train pairs 推出不變量（輸出形狀規則、顏色集合、顏色守恆、對稱等），對 E1 HA 每個 `task_i` 的完整 kgmon 排序清單做**穩定過濾**：違反不變量的候選移出，其餘順序不變；剩不到 2 個就用原順序補回。執行時涵蓋全部 120 題 → **c＝1**。
- **F（C1 第 1 部分的部分替代；這是輸出過濾，不是等變性測試）**：同一組不變量套在 LLM 程式的測試答案上，違反的答案不進 merge。只影響程式通道 → **c＝0.65**。
- **D（ARC-1）clean-room CPU DSL**：對所有 train pairs 全對的 DSL 程式，其測試答案跟 LLM 程式答案一樣進 `v13_merge.merge` 的 `results`。CPU 在所有題目上執行 → **c＝1**（要另外確認 Kaggle CPU 時間預算）。是否建置，等 R＋F 結果出來、看剩下差距再決定（不影響 D 的設計）。
- progsynth.feedback 修補與 NVARC 票數差門控：Argon 已指出在 E3 靜態 log 上無法生效（lost_correct 0/40、12/12 自然結束已通過驗證）→ 本重播不計。

## 2. 公式（寫死）
- 格子＝E3 rep {1, 2} × E1 HA 種子 {1, 2, 3}＝6 格（與 v2 的 8.278／7.753 相同）。分數＝keys30 題分總和（每題＝答對輸出數／輸出數）。
- 每格：`S_base`＝R 過濾後的 HA top-2；`S_D`＝S_base ⊕ D；`S_all`＝S_D ⊕ (F 過濾後的 E3 該 rep 程式答案)。⊕＝`v13_merge.merge`，不改規則。
- **adj＝mean₆[ S_D + 0.65 × (S_all − S_D) ]**。
- 健全性檢查：R、F、D 全關時，必須重現 HA 6.778、raw 8.278、adj 7.753（差 >0.001 就先修 harness，不看結果）。

## 3. 閘門（寫死）
- **通過＝adj ≥ 8.253**（v2 7.753＋0.5），而且：
  - R 沒有把任何一格中本來在 NVARC top-2 且正確的答案移出 top-2；
  - F 沒有拒絕任何正確的 LLM 答案（E3 rep 0/1/2、E6、v2、v3、v4 全部列出檢查）；
  - D 沒有因錯誤答案把正確的 NVARC 答案擠掉（lost_correct＝0）。
- 只供參考、不當閘門：E6（96k，c96＝0.37）的同樣計算；D 在評估集其餘 90 題上解出的題數。
- **不通過** → ARC-1／C1 不花 GPU；C3／C2 探測照自己的閘門另行評估；三者都不過 → 0 GPU，鎖定 57165541＋56864814，10/29 12:00Z 前釋出份額。
- **通過** → 最多一個綁定 kernel（≤13.8k GPU-s，R1 自動閘門＋kaggle_env_lint＋第一分鐘備援 submission＋存完整程式）。

## 4. 防過擬合（寫死，先於任何 keys30 結果）
- 不變量與 DSL **只用** `arc-agi_training_challenges.json`（1,000 題）及評估集**非 keys30** 的 90 題設計與除錯。設計期間不打開、不列印 keys30 的題目或解答。
- 不變量的健全度先在這 1,090 題量：由 train pairs 推出的不變量，正確測試輸出必須滿足；每條不變量的誤殺率 ≤1%，否則刪掉那條（在跑 keys30 前決定）。
- keys30 重播每個凍結版本只跑一次，跑前記下程式 sha256。看到 keys30 結果後只准修當機類錯誤，而且兩個版本的數字都要報；不准依 keys30 結果調不變量或 DSL。
- clean-room：不讀、不執行公開 notebook 的 `symbolic_loo.py`。

## 5. 分工（R7／R8）
- 機械部分（store 讀取包裝、重播 harness、測試）交 DeepSeek `ds_agent.py --edit`；DSL 草稿交 Argon `gemini_api/ask.py --mode write`。我設計、套用、跑測試、讀每份 diff。

## 結果 1：R＋F（2026-10-19 07:1xZ；凍結 commit 3530121f，invariants b07f130d、replay 67e6b67b）
- 不變量健全度（training 1,000＋評估非 keys30 90 題，`arc1-soundness-v1.json`）：8 條 ≤1% 保留（colors_subset_in、colors_subset_in_plus、hist_eq、keep_nonbg、palette_eq、shape_same、shape_transpose、sym_transpose）；6 條 >1% 刪除（shape_const 4.7%、shape_ratio、subgrid_of_in、sym_flip_lr／ud／rot180）。中間只調過一次（shape_const／sym／subgrid 要 ≥3 個 train pair），只看開發集。
- 全關健全性：HA 6.778、raw 8.278、adj 7.753、E6 參考 7.837 → 完全重現（`arc1-replay-sanity-alloff.json`）。
- **R＋F 開（`arc1-replay-RF-v1.json`）：adj 7.753，與 v2 完全相同**。R 救回 0、誤殺 0；F 對所有 log 的驗證答案拒絕 0（含錯誤答案 0）。
- 診斷（scratchpad，非閘門）：E1 HA 每個輸出的 kgmon 池平均只有 3.6 個不同候選；534 個候選只有 4 個違反不變量；正解不在池中 112／150，排第 3–5 名 7 次（3 種子合計）→ 池重排的上限本來就小，而且 NVARC 的錯誤答案幾乎都符合形狀／顏色不變量。
- **結論**：C1（池重排＋答案過濾）對閘門貢獻 0。距 8.253 的 +0.5 只能靠 D（DSL，c＝1），約需在 keys30 多解出 1 題 NVARC 沒解的題。

## 補充預登記：D（DSL）（2026-10-19 07:1xZ，DSL 草稿還不存在時寫）
- **開發尺標**：training 1,000 題＋評估非 keys30 90 題（keys30 用 `soundness.py` 的同一方式跳過，讀任何欄位前就跳過）。報：解出題數（所有測試輸出全對）、**精確度**＝train 全對的 DSL 程式中測試也對的比例。
- **精確度規則（寫死）**：同一題若有兩個以上 train 全對的 DSL 程式而且測試答案不同 → 該題視為未驗證、不出答案；只有唯一答案（或所有驗證程式答案一致）才出。若開發集精確度 <0.75，再加一條「排除恆等／只換色而輸出等於輸入」等退化程式，在跑 keys30 前定案。
- **CPU 預算**：每題 ≤10 s（單核），120 題總計 ≤20 min；超時的題不出答案。開發集上量 p95 時間。
- **clean-room**：Argon 只拿到規格與 grid 格式說明，不拿任何題目 JSON；我在本機跑。
- **閘門不變**：keys30 跑一次，adj ≥8.253＋lost_correct＝0。換算：HA 漏掉的 keys30 題，DSL 解出一題、LLM 也沒解 → 約 +1.0；LLM 兩個 rep 都解 → 約 +0.35。實際上要多一題新解。
- **背景數字**：Argon P（D 過閘門）≈12%。

## 結果 2：D（DSL）＋最終判定（2026-10-19 07:3xZ）
- DSL＝Argon 依 `arc1/dsl-spec-v1.md` 寫的草稿，未修改（`arc1/dsl.py` sha256 9ab266d9…，1,625 行，只用 stdlib＋numpy；Argon 自己的第 1 個玩具測試本身不一致，未採用）。runner `arc1/run_dsl.py` 7b14ee4a…，每題 10 s，4 個行程（白天只做輕量檢查，`configs/local-compute-approval-v1.json`）。
- 開發集：評估非 keys30 90 題 **0 題有任何 train 全對的程式**（`arc1-dsl-dev-eval90-v1.json`）；training 前 100 題解出 8、精確度 1.0、6 題因程式答案不一致而不出（`arc1-dsl-dev-train100-v1.json`）→ 求解器本身會動，只是 ARC-AGI-2 評估題超出它的範圍。p95 10.1 s（都跑滿時間上限）。
- keys30（只跑一次）：0／30 驗證（`arc1-dsl-keys30-v1.jsonl`）。R＋F＋D 重播（`arc1-replay-RFD-v1.json`）：**adj 7.753**，lost_correct 0、F 誤殺 0。
- **判定：不通過（7.753 <8.253）→ ARC-1／C1 不花 GPU。** C3 已紙面淘汰（ARC-AGI-2 semi-private 1.67%）。剩下只有 C2 沒有便宜的離線測量（見 HANDOFF）。
- **修正（07:4xZ，advisor）**：runner 原本 `verified` 要求 dt ≤10.0 s，但求解器跑滿 10 s 上限、p95 10.09 s → 可能把唯一答案的題默默標成未驗證。改成 dt ≤ LIMIT＋1（求解器自己有期限，LIMIT＋5 硬殺），並記 `found_unverified`。用同一個 dsl.py（9ab266d9）重跑：評估非 keys30 90 題 verified 0、found_unverified 0（`arc1-dsl-dev-eval90-v1b.json`）；keys30 30 題全部 n_programs 0（`arc1-dsl-keys30-v1b.jsonl`）→ **結果 2 不變**。
