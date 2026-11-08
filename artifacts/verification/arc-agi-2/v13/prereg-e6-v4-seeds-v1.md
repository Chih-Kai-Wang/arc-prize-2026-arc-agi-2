# E6 預登記：v4-c96 程式通道兩種子離線量測（2026-10-17 ~18:30Z 寫定；推送前可改、推送後不改）

## 目的
- v4-c96（cap 96k、無強制；提交 57301011 評分中）依 `prereg-v4-v1.md` §2 **最多當第 2 名額**，除非之後有多次離線重複（`final_selection_amendment_2026_10_13`）。
- E6 就是那個多次離線重複。它與 E3（v13 v2 的兩種子量測）同設計、同種子，所以兩者可以配對比較。
- 時機：使用者 10/18 06:00Z 起停工，可能好幾天。E6 在停工期間於 Kaggle 上跑完，回來後只做對帳與重播（零 GPU）。本週 reserve 不用就在 10/24 00:00Z 失效。

## 設計（`runs/arc-agi-2/v13/build_e6.py`，kernel `arc-2026-v13-e6-v4seeds-v1`，Kaggle 4×L4，commit 跑，不是提交）
- 程式碼＝`src3/`（與 E3／E4／v4-c96 同檔），呼叫 `v13_llm.py --jobs e6-jobs.json --cap 96000`：
  - 無 `--force-at`，即 v2 行為。
  - 併發照 KV 公式：KV 331,200 → 10，與 v4-c96 相同。
  - 提示、增強、驗證、防記憶檢查都與 v4-c96 相同。
- 不跑 NVARC，只有 LLM 階段。任務＝公開 eval 的 keys30，以 eval 解答評分。notebook 在 rerun 會 assert 拒跑。
- 工作清單（`e6-jobs.json`，sha256 29c0eee2…）：
  - keys30 依序交錯 (題, rep 1)、(題, rep 2)，共 60 個。
  - **與 E3 的種子工作完全相同**：同 rep，所以種子與增強 salt 都相同。tests_e6 斷言 E6.jobs() 等於 E3 的種子工作。
  - 沒有 rep 0 尾段。
- 牆鐘 `E6_SECONDS`＝35,000：
  - setup 約 650 s。
  - 生成：60 個工作 ÷ 併發 10＝6 波，每波 ≤5,500 s，合計 ≤33,000 s。依據：v4-c96 dev run 的 96k 工作在約 5,300 s 時於 89–92k 被中止；E4 在 >64k 的解碼速度中位數 18.55 tok/s。
  - 中止餘裕 360 s。
  - 規劃牆鐘 35,600（含開機 600）< Kaggle 12 h（tests_e6 斷言）。
  - 規劃扣額 35,600 × 2.2＝**78,320**。E3 的實扣是規劃的 0.89，所以預期實扣約 70k。

## 主估計（預先寫定，結果出來後不改）
1. 對 r ∈ {1, 2}：
   ```
   .venv/bin/python runs/arc-agi-2/v13/replay_v2_merge.py runs/arc-agi-2/gcp-e1/out/e1 runs/arc-agi-2/v6/output-arc-2026-v13-e6-v4seeds-v1/v13-programs.jsonl artifacts/verification/arc-agi-2/v13/replay-e6-rep{r}.json --rep {r}
   ```
   得到每個 E1 HA 種子 s ∈ {1, 2, 3} 的 after 分數。
2. **v4 原始離線均值＝2 rep × 3 HA＝6 格的平均**。同時報：
   - 6 格 SD；
   - 每個 rep 的值；
   - 每格相對 HA 的增量；
   - `lost_correct_nvarc_any`；
   - 被 E6 期限中止的種子工作數。
3. **與 E3 配對**：同 (rep, HA 種子) 的 6 個配對差，即 E6 格 − E3 格（E3 檔＝`replay-e3-rep{1,2}.json`），並報平均與 SD。
4. **覆蓋率調整**（候選表與第 1 名額比較用這個）：
   - 公式：adj＝HA 均值 6.778 ＋ c ×（原始均值 − 6.778），與 E3 同。
   - v2：c64＝0.65 → 7.753（已寫定）。
   - v4：**c96＝0.65 × jobs96／jobs64＝0.65 × 42.1／74.5＝0.37**。jobs 用 `cap-yield-v1.json` stratified 的 `plan_strat`：64k-c13 有 74.5 個；96k-c10 在 E4 的 q≈0.4 時約 42.1 個，由 q 0.3 的 41.4 依斜率外推。
   - 敏感度：c96 ∈ {0.30, 0.45}。
5. **中止規則（同 E3）**：被期限中止的種子工作 >8 個 → 主估計標記「不完整、下界」，候選表註明，不補跑。

## 怎麼用（回來之後；停工期間不做任何決定）
v4 成為**第 1 名額候選**，要下列全部成立：
1. adj(v4) − 7.753 ≥ **0.5**（離線持平帶；c96＝0.37）。
2. 敏感度 c96＝0.30 時，差值仍 ≥ 0。
3. 中止的種子工作 ≤ 8。
4. 6 格都沒有 `lost_correct_nvarc`；若有，記錄並視為不成立。
5. v4 公開分 ≥ 32.22 − 2.4＝**29.82**。v4 未出分或 ERROR → 不成立。

不成立時：v4 維持「最多第 2 名額」；第 2 名額照 slot2 規則（v8）。

成立時：照最終選定流程改選，也就是在對話問使用者一句、Fable 審查、回報協調者。**v4 的公開分本身永遠不觸發改選。**

報告（不設門檻）：配對差平均、每 rep 差、96k 自然結束數與全對數，與 E4 對照。

## 綁定（推送前；任何一項變了就重建、重審）
- notebook c1797eae…、kernel-metadata 199c83d5…、e6-jobs 29c0eee2…；src3 各檔 sha 記在 `preparation.json`，與 E3／E4／v4-c96 相同。
- build_e6.py 33fc7860、launch_v13.py 100c2d74（新增 gate_e6，其餘不變）、tests_e6/test_e6.py 7abd7dc0。
- 共用佇列 `scripts/kaggle_gpu_queue.py` **fe170603**（patch v3，main ≥ e2e5b27d）。
- 後處理：replay_v2_merge 8f3dda2b、cloud 6cb9fe14、robust_download 728b3ba5。
- 檢查：
  - tests_e6 全過：工作清單＝E3 種子工作、notebook 呼叫、假 server 上 cap 96k 每個工作只跑一次、併發 10、launch gate 接受本建置並拒絕被改的 notebook。
  - tests_e4 仍過。
  - `kaggle_env_lint.py` 0 RISK。

## 推送條件（全部成立才推；任何一項不成立＝不推、記錄）
1. 本預登記＋E6 發射檔的 T3 審查：Argon＋Opus 平行盲審，`run_tiered_review.py --gpu-launch`。逐條核對 P0／P1，advisor 裁決 GO。
2. 協調者把 reserve 撥給 ARC（`configs/resource-allocation-v3.json` 本週 ARC 份額 ≥ 已用 39,694＋78,320），合併含該 commit 的 main 後 `launch_v13.py --gate-only` 仍通過。
3. 容量預查（唯讀）：jsonl 近 24 h 內 pushed=true、未 settle 的非 TPU kernel 逐一 `kaggle kernels status`，QUEUED／RUNNING <2。沒有在跑的 ARC kernel（launch 本身也會查）。
4. **最晚 10/18 03:00Z 推送**；超過就不推，寫進 PAUSE BLOCK，作為回來後的第一個選項。
5. 推送被拒（例如 2 個 session 上限）：照 10-17 規則，不重推、settle 0（`settle_unpushed`）、記錄。

## 回來之後的步驟（依序）
1. `cloud.py status arc-2026-v13-e6-v4seeds-v1`。
2. settle：`settle_charge(<reserved_at>, <charged>, source=…, measured_at=<結束時的額度讀數>)`；重疊被拒就交協調者。
3. `robust_download.py <worktree> arc-2026-v13-e6-v4seeds-v1`。
4. 照上方「主估計」步驟 1–5 計算。
5. 更新候選表。
6. 照「怎麼用」判定；改選只走閘門。

## 推送前修正（2026-10-17 ~19:0xZ，T3 審查 Opus GO-WITH-FIXES＋Argon GO-WITH-FIXES 之後；notebook 不重建，發射檔不變）
審查：`reviews/20261017T182407Z-e6-prelaunch-{opus,argon}.md`；摘要 `api-reviews/20261018-e6-prelaunch-summary.md`。

### 1. 「沒跑完」改為明確計數（取代主估計第 5 點；Opus F2／F3，Argon P1-2）
- 依 `e6-jobs.json` 的 60 個種子工作，對照 `v13-programs.jsonl` 逐一分類：
  - **完成**：`state`＝done，或 `state`＝aborted 且 `completion_tokens` ≥ 90,000（已接近上限，是期限造成的）。
  - **未完成**：沒有列（從未開始）；`state`＝exception；`state`＝aborted 且 `completion_tokens` < 90,000。
- 未完成 > 8 → 主估計標記「不完整、下界」。
- 下界的處理（Argon P1-2）：未完成的工作只會少解題，所以下界偏低。**下界若仍通過「怎麼用」的條件 1，照樣算成立**；沒通過就是不成立（不補跑）。
- **伺服器故障**：`state`＝exception 的種子工作 > 8 → 標記「伺服器故障、不可比」，不用於第 1 名額（不論數值）。

### 2. 可比性守門（Opus F6）
- `v13-llm.json` 的 `conc_effective` ≠ 10 → 不可比，不用於第 1 名額；`cap` ≠ 96000 → 同樣處理。
- 伺服器實際選用的設定不是 MTP 2 → 同樣處理（這和 v4-c96 dev run 的設定不同）。

### 3. 配對的意思（Opus F4，只是說明）
- 同 (題, rep) 只保證增強與種子相同。E3 是併發 13，E6 是併發 10，加上 FP8、MTP、批次數值差異，生成軌跡幾乎一定會分岔。
- 所以「E6 − E3」只消掉增強的變異，**不能**解讀成「同一段推理在 64k 之後繼續」。
- 報告時以 **每 rep 的差** 為主要誤差範圍（程式通道的有效 n＝2），6 格 SD 只當參考。

### 4. 門檻維持 0.5（Argon P1-1 部分不採納）
- Argon 指出兩件事：(i) 條件 2（c＝0.30 時 ≥0）在條件 1 成立時必然成立，是多餘的；(ii) E4 的中心預估大約是 +0.36，會卡在 0.5 下。
- (i) 正確，但無害；條件 2 保留，只是報告用。
- (ii) 0.5 是第 1 名額改選用的**離線持平帶**，寫在 final_selection 規則與 10/13 雜訊帶，所有候選都適用，不能為了 v4 改。真實增益若只有 +0.36，依規則就不該換第 1 名額。
- 線性 c 偏保守（unsure-first 排序讓覆蓋的前 42 題價值較高）：記錄為已知偏差，不改。
- **另外加一項報告（不設門檻）**：E6 − E3 配對原始差 ≥ +1.5 題時，在候選表註明「96k 的原始效果大」，供回來後設計下一版（例如分層上限）參考。

### 5. 共用佇列與 settle（Opus F7、Argon P2-3／P2-4）
- 唯讀模擬 `gpu_launch`：ARC spent 39,694.361 ＋ 78,320＝118,014.361 > 份額 118,014，reserve 0 → 會拒絕 0.361 s。已請協調者把份額改為 **118,015**。推送條件 2 改成「份額 ≥ spent ＋ 78,320，以 `gpu_launch` 的同一算法計算」。
- **settle 預先授權**（停工期間沒有人在結束時讀額度）：協調者在 E6 結束後的心跳，以 `settle_charge(<reserved_at>, 2 × kernel 經過秒數, source=…, allow_overlap=True)` 結算。經過秒數取自 log 最後時間戳（同 v4 的做法）。回來後由本對話核對。
- 回來之後的步驟 2 依此修改：先查協調者是否已 settle；沒有就照上述算法 settle。

### 6. 2 個 session 上限（Argon P2-5）
- `launch_v13.py` 只查 ARC 自己的 kernel。推送前照推送條件 3 手動查 Gemma／Enveda 的 kernel。
- 被拒照推送條件 5：不重推、`settle_unpushed`、記錄。協調者也要求 E6 只佔 1 個 session。

### 7. delta 審查後的釐清（Argon T1 delta GO，P2-1；推送前，只加嚴）
- 「未完成」＝60 − 完成。凡不是 done、也不是「aborted 且 ≥90,000」的狀態都算未完成，包括 `http_error`、`incomplete`、`job_exception`、`completion_tokens` 為 null。
- 「伺服器故障」守門改計**所有錯誤狀態**（exception、http_error、job_exception、incomplete）合計 > 8。

## 執行中的註記（2026-10-17 21:5xZ，E6 RUNNING、還沒有任何輸出列；只改用途，不改計算）
- v4-c96 的提交 57301011 公開分 **29.03**（21:50Z 出分），比 v2 的 32.22 低 3.19pp，超過 2.4pp 雜訊帶。依「怎麼用」**條件 5（≥29.82）不成立** → **E6 不論結果都不用於第 1 名額**，最終選定維持 57165541＋56864814。
- 主估計、守門、配對報告照寫定方式計算。
- **新增描述用診斷（不設門檻）**：
  - 96k 自然結束工作的驗證精確度與全對精確度；
  - 每格的 `lost_correct_nvarc`，以及被擠掉的題；
  - 與 E3 同題的配對差。
- 這些用來判讀 v4 公開分為什麼跌 3.19pp，並當 10/24 週設計的輸入。判讀的兩個方向：96k 程式的偽陽性擠掉了正確的 NVARC 答案？還是覆蓋率下降？
