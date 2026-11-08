# 預登記：C2 測量 kernel（低 effort＋真執行修補）（2026-10-19 07:4xZ，寫在任何 C2 程式之前）

來源：協調者 10-19 ~07:4xZ 核准一個 ≤13,800 GPU-s 的 C2 測量 kernel（`configs/resource-allocation-v3.json` reserve_grant_2026_10_19_arc_c2；10/24 00:00Z 以後推送則改扣 10/24 週份額）；Argon `newdir-arc-agi-2-20261019/argon-r1.md` §C2；R1 自動閘門；R2 可綁多項。

## 1. 設計（src6＝src〔v2〕＋下列改動；NVARC 與 merge 規則不變）
- **effort／上限依 rep 排程**：rep 0＝`medium`、首輪上限 16,000 tokens；rep ≥1＝`high`、上限 32,000。佇列規則照舊（先跑每題 rep 0；沒驗證的題才排下一個 rep，MAX_REPS 6）。
- **真執行修補**：一個工作的候選（content 最後一段程式、reasoning 最後兩段）都沒驗證，而且至少有一段程式碼時，取 train_pass 最高者（同分取 content），用 `progsynth.feedback` 的真實執行結果再問一輪；最多 2 輪，每輪上限 8,000 tokens，effort 同該工作。完全沒有程式碼的工作不追問（避免 v3 的強迫作答）。驗證與 `suspicious` 守門不變。
- **存完整程式**：驗證程式與最佳未驗證候選存完整原始碼（≤20,000 字）＋修補輪數與 tokens。
- **併發**：沿用 KV 公式，A.cap 取排程最大值 32,000（v2 在 64k 時得 13 → 預期約 20）。
- **R1 閘門**：隱藏資料形狀測試（不同 ID／數量／順序）、kaggle_env_lint、CPU 編譯／tests、**提交 kernel 第一分鐘先寫備援 submission**、不寫死公開測試的雜湊／ID／數量。
- commit（dev）run＝與 v2 相同的 7,200 s：NVARC 煙霧測試＋keys30 程式工作。預估約 13.8k GPU-s。

## 2. 測量公式（寫死）
- 隱藏集程式通道視窗 W＝23,300 s／120 題（`design-1024-paper-screen-v1.md` 的 c 定義）→ 每題等效時間＝dev run 生成時鐘 **t\*＝23,300 × 30／120＝5,825 s**。
- 取 dev run 中 `t_end ≤ gen_start＋5,825 s` 的工作（dev 生成時間不足 5,825 s 就全取，保守）。
- 每個 HA 種子 s∈{1,2,3}：S_s＝keys30 分數（E1 HA-s top-2 ⊕ 這些工作的驗證答案，`v13_merge.merge`）。**adj_C2＝mean_s S_s**（時間模擬已含覆蓋率，不再乘 c）。
- 比較基準：v2 adj **7.753**（不變）。
- 逐題配對：v2 每題＝HA 題分＋0.65×(E3 rep1／rep2 平均題增益)；C2 每題＝上述 S 的題分（3 種子平均）。Δ̄＝adj_C2 − 7.753。對 30 題做配對 bootstrap（10,000 次）→ P(Δ>0)。
- 另報（只供參考）：rep 0（medium）與 rep ≥1（high）各自的自然結束率、驗證率、精確度、每工作 tokens；修補輪觸發數與救回數；吞吐（tokens／s、工作／s）；lost_correct。

## 3. dev run 稽核（自動，決定要不要送那一筆探索提交）
- effort 生效：rep 0 完成 tokens 中位數 <16,000；排程照跑（有 rep ≥1 的 high 工作）。
- 修補迴圈至少觸發 1 次；完整程式有存；無搶占（`vllm:num_preemptions_total` 0）；KV 峰值 ≤0.95；工作例外 0。
- 任一不過 → 不提交，算一次測量並記錄。

## 4. 進最終選定的條件（寫死；公開分只是一個雜訊樣本）
- **第 1 名額**：Δ̄ ≥ **+0.5** 而且 P(Δ>0) ≥ **0.8** 而且探索提交公開分 ≥ **29.82**（v2 32.22 − 2.4，可靠性否決）。三者都成立才提案改選；改選仍要在對話問使用者一句＋T3＋Fable＋回報協調者（截止 11/02）。
- **第 2 名額**：維持 v8 56864814（R4「與第 1 名額最不相似」：C2 與 v2 共用 NVARC 底與 LLM 通道，v8 才是最不相似者）。
- 其餘情形：C2 只算資訊，最終選定維持 **57165541＋56864814**。
- 只有一個 LLM 種子（dev run）：E3 rep 間原始分差 2.67（單 rep SD ≈1.9），所以 P(Δ>0) 這條是必要的，不能只看 Δ̄。

## 5. 停損
- Claude 週額度在推送前達 ≥85% → 停，維持預設（11/02 結案）。
- 建置／測試卡住超過一天 → 停，釋出 13,800。
- 只推一次；不重推、不加第二格。

## 6. 分工
- 程式改動由 DeepSeek `ds_agent.py --edit`（附我寫的測試指令）起草；Argon 寫草稿；我設計、套用、讀 diff、跑測試、審發射路徑。

## 修正（2026-10-19 07:5xZ，advisor；仍在任何 C2 程式之前）
1. **effort 值（取代 §1 的排程）**：官方 chat template（https://huggingface.co/Qwen/Qwen3.8-27B-FP8/raw/main/chat_template.jinja）只接受 `xhigh`（預設）、`medium`、`low`，其他值 `raise_exception` → 原排程的 `high` 會讓每個 rep ≥1 工作都失敗。`medium` 不注入任何指示（＝模型預設思考）；P0 的 effort_check（xhigh 1,863 vs low 1,851 prompt tokens）與此一致。**新排程：所有 rep 都用 `medium`；rep 0 上限 20,000、rep ≥1 上限 32,000**；修補輪同樣 medium、8,000。§3 的「rep 0 中位數 <16,000」改為「<20,000」。
2. **kernel 內 effort 閘門**：`smoke()` 對排程中每個 effort 值各送一個 1-token 請求，非 200 就 `Skip`（損失限於 setup 成本）。推送前也用 template 原文核對（上面的 URL）。
3. **同協定對照組**：把同樣的 t\*＝5,825 s 截斷套用到 v2 自己的 dev run（`output-arc-2026-v13-progsynth-v2/v13-programs.jsonl`）→ `v2_dev_adj`，當次要、同條件的比較；**決策基準仍是 7.753**。預期 dev 生成階段約 5,000 s（7,200 減 NVARC 煙霧、安裝約 250 s、載入 server、停派／中止邊界），「不足就全取」會讓 C2 少算約 G／5,825 ≈15%；主數字維持保守，按比例放大的數字只當資訊。
4. **第一分鐘備援**：新增 notebook cell，在 rerun 一開始用 `arc_v6_config.fallback_submission(test)` 寫 `submission.json`。已核對 `assemble.py:130` 無條件覆寫 `submission.json`、`starter.py` 不讀它 → 安全；NVARC 檔仍與 v11 位元相同。
5. **（07:5xZ，仍在任何 C2 結果之前）同協定對照組的數字已算出、第 1 名額條件加嚴**：`c2_measure.py` 拿 v2 自己的 dev run 自我測試：v2_adj_check 7.753（重現）；**v2_dev_adj＝8.778**（生成 4,864 s <5,825，26 列全取，4 筆驗證）→ 對 7.753 的 Δ̄＝+1.03、P(Δ>0)＝0.785。也就是說，一個運氣好的 v2 種子在這個協定下就幾乎通過原條件。三個 LLM 種子的增益（E3 rep1 +0.17、rep2 +2.83、v2 dev +2.0）SD ≈1.4，題目 bootstrap 抓不到這個種子變異。**第 1 名額改為四個都要成立**：Δ̄ ≥+0.5（對 7.753）、P(Δ>0) ≥0.8、**adj_C2 ≥ v2_dev_adj＋0.5＝9.278**（同協定、同視窗、各一個種子）、公開 ≥29.82。其餘不變。按比例放大的資訊數字改為 HA＋(5,825／G)×(adj−HA)（原本誤寫成上限 1）。
6. audit 另保留 v2 的出貨規則（keys30 至少一筆驗證且正確、精確度 ≥0.5）當探索提交的附加閘門；R1 下不需獨立審查（audit 對 C2 不檢查 review）。
7. **（08:3xZ，Argon 推送前檢查後；仍在推送之前）** 審查紀錄 `c2-argon/prepush-review-v1.md`：
   - P0（採納）：「rep 0 中位數 <20,000」會把「medium 思考超過 20k」這種**測量結果**當成 harness 失敗而擋掉提交。改成閘門「rep 0 上限有生效（max ≤20,000）」＋「有 rep ≥1 工作」；中位數只記錄。
   - P1 修補歷史重複（採納）：第 2 輪改為附上第 1 輪最好的新程式與其回饋；沒有新程式就停。P1 被守門拒絕的程式（train 全對）不再當修補起點（採納）。P1 HTTP 錯誤時 t_end 為 None（採納，補上 t_end，修補輪不覆蓋）。
   - P1 commit 7,000 vs 預登記的 7,200（採納為修正）：為了 13,800 額度，C2 的 commit 是 7,000 s（規劃牆鐘 7,600）。→ v2 dev 對照組改用**同一個生成時間上限**（min(5,825, C2 觀察到的生成時間)）截斷，比較才同條件。
   - P1 KV（駁回）：伺服器回報的 kv_tokens 331,200 不是實際容量；HANDOFF 的有效 KV ≈1.2M（R² 0.993），v2／E3 在 64k、13 條時峰值 0.73。C2 公式得 conc 21，最壞 21×約 40k ≈840k ≈0.70。
   - P2（採納）：audit 加「完整程式也存在未驗證的最佳候選」；修補前再查一次 STOP／abort；summarize 的 from_reasoning 計數含修補來源。
   - 重新建置為 `arc-2026-v13-c2-medrep-v2`（v1 從未推送，作廢）。
8. **（08:3xZ，推送前；audit 不在 kernel 內）** rep 0 上限閘門容許 +8 tokens（MTP＝2 的 completion_tokens 可能略超 max_tokens）；未驗證列的 best_code 只要求「存在」（空程式區塊不誤判）。C2 的 audit 不讀 `--review`（R1），呼叫時傳本預登記路徑。已知風險（不是 harness 錯誤）：修補輪也開 thinking，8k 上限可能全被思考用完而沒有程式 → 該輪 n_code 0、迴圈結束；這是 C2 的測量結果。

## 結果（2026-10-19 10:4xZ）：不通過 → 不進最終選定、不提交
- kernel `arc-2026-v13-c2-medrep-v2` v1：08:34Z 推送 → 10:34Z COMPLETE；**settle 13,348.2 GPU-s**（額度 13,800 內）。下載 65 檔到 `runs/arc-agi-2/v6/output-arc-2026-v13-c2-medrep-v2/`。
- audit（`v6/audit-arc-2026-v13-c2-medrep-v2.json`）：C2 閘門全過（排程／medium 生效、effort_smoke ok、rep 0 上限生效、有 rep ≥1、修補有觸發、完整程式有存、搶占 0、KV 峰值 0.673、conc 21、無例外）；**v2 出貨規則的 precision_floor 不過**（驗證 3、正確 1，精確度 0.33 <0.5）→ `submission_allowed` false → **不送探索提交**。
- 量測（`v13/c2-measure-v1.json`）：生成 5,149 s（<5,825，83 列全取）。**adj_C2＝6.778（＝HA，程式通道淨增益 0）**；Δ̄＝**−0.975**、P(Δ>0)＝**0.0**；同協定 v2_dev（截在 5,149 s）＝8.778。四條件全不成立。
- 機制層面：
  - medium（不注入指示）**並沒有縮短思考**：rep 0 34 個工作 33 個撞 20k 上限、只有 3 個有程式碼；rep ≥1（32k）49 個中 9 個自然結束、5 個有程式碼。
  - 修補觸發 6 次：5 次在 8k 內全部用來思考、沒有程式碼；1 次產出驗證程式但測試錯（28a6681f，偽陽性）。
  - 驗證 3 筆：38007db0（rep 0、reasoning 草稿、正確，NVARC 本來就有）、53fb4810（已知偽陽性題）、28a6681f（修補產生的偽陽性）。
  - 吞吐：tokens 2.04M（v2 dev 1.17M，×1.75；conc 21 vs 13），但每個工作的產出遠低於 xhigh。
- **結論**：C2（effort 降低＋真執行修補）在這個模型上不成立——medium 不會更早寫出程式，修補輪也被思考吃掉。最終選定維持 **57165541＋56864814**。
