# v13 dev run 預登記（2026-10-13，推送前寫定）

## 是什麼
- kernel：`f0909172434/arc-2026-v13-progsynth-v1`（`runs/arc-agi-2/v13/build.py`；啟動 `runs/arc-agi-2/v13/launch_v13.py`）。commit 跑＝開發量測；同一版本就是提交候選（通過就直接可送，不再推第二次）。
- 流程：v11 NVARC（6 個檔案與 `runs/arc-agi-2/v11/src` 逐位元組相同；`ARC_V8_PASS_B_MIN_SECONDS=1e12` 關閉 pass B）→ `assemble.main()`（NVARC 的 submission）→ `v13_llm.py`（Python 3.12.14 standalone venv＋vLLM 0.19.0 wheelhouse；Qwen3.8-27B-FP8；TP4、MTP2、async、conc 32、max-model-len 131072、xhigh）→ `v13_merge.finalize`（只在 rerun）。
- 規劃牆鐘：commit 5,400 s＋開機 600 s → 規劃扣額 6,000×2.2＝**13,200**（本週剩約 20.2k；10/17 00Z 前跑）。

## 預先決定的設計（不因 dev run 結果事後改動，除非整合失敗）
- 填答（`v13_merge.py`）：每個 test output，驗證通過程式的答案（多數決，同票取最先通過者）→ attempt_1；NVARC top-1 → attempt_2；程式答案已等於 NVARC 兩個答案之一 → 不改；NVARC 沒有真答案 → attempt_2 用下一個不同的程式答案。
- 驗證：在**增強後**（顏色置換＋二面體）的全部訓練對上完全重現，且每個 test 輸出都是合法格子；答案反變換回原座標。
- 候選程式：最終回答裡最後一個 `def transform` 區塊；未通過、截斷（length）或到期中止時，再試推理文字中最後兩個程式草稿（零額外 token；靠訓練對驗證守精確度）。
- 防記憶（API 審查 #12）：通過驗證但①原始碼含某個訓練輸出（≥9 格、≥2 色）的數字序列、②test 答案等於某個訓練輸出（≥9 格）、③訓練對都不是恆等但 test 答案＝test 輸入 → 拒收。
- 併發：server 起來後依 KV 容量調整 `conc = max(8, min(32, int(kv/(0.5·cap+6000))+2))`（4×L4、kv≈344k、cap 64k → 約 11；commit cap 20k → 約 23），避免長上下文時 32 條互相搶占。
- 模型雜湊：只在 commit 背景計算、只記錄不阻擋（P1 已驗過這個釘住版本；程式答案另有訓練對驗證把關）。
- 排序（rerun）：NVARC pass-A kgmon 信心由低到高（無候選＝−inf 最先），信心 ≥5 的題放最後；每題先跑 1 個 rep，未通過才把下一個 rep（換增強）排到隊尾；不做修補輪。依據：keys30 上程式獨贏的兩題（f931b4a8、dbff022c）HA 信心約 −4。
- 時間：rerun 的 LLM 階段在 NVARC 自然跑完、assemble 後才開始；不縮短 NVARC 期限（per-task 上限與 v11 相同 → E1 的 HA 數字可沿用）。結束前 21 分停止派新工作、前 6 分中止進行中的串流並試其推理草稿。
- commit 模式：任務＝NVARC 冒煙 4 題（同排序規則）＋keys30（公開 eval，清單順序）；`max_tokens` 20,000（讓工作在 ~40 分鐘內真的跑完、走完驗證→合併→寫檔）；以 eval 解答評分（只當量測，不當決策的主證據）。
- 失敗即降級：時間不足、輸入缺、GPU 未釋放（5 分＋殺孤兒程序後 2 分）、wheel／Python／模型雜湊不符、安裝失敗、server 起不來 → 跳過 LLM 階段，保留 NVARC submission。

## 整合通過（pass/fail）＝以下全部成立
1. NVARC 冒煙完成且 assemble 正常（`v6-assembly.json` 無 error）。
2. LLM 階段 `skipped` 為 null：GPU 已釋放、venv 安裝＋import（cuda_ok）通過、server ready＋smoke 通過。
3. 至少 1 個工作完成（有 finish_reason 或 aborted 且已嘗試驗證），`v13-programs.jsonl` 有紀錄。
4. 若有驗證通過的程式：`eval-v13-submission.json` 存在且 `validate` 通過；`submission.json`（test 檔）仍是合法備援。
5. 核心時間：LLM 階段在 `v13_end_time` 前返回，notebook 最後一格 `v13_done` 有印出。

## 要讀的量測（記入 HANDOFF，不設門檻）
- 安裝秒數、server 啟動秒數、KV token 數、生成期間 tok/s 與 preemption（`v13-metrics.jsonl`）。
- keys30：完成數（stop／length／aborted）、驗證通過數、其中正確數；來自推理草稿的通過與正確數。
- 與 P0b（Colab 單卡、64k 上限）比較只作參考：上限不同（20k）。

## 送出決策規則（預先寫定）
- **送 v13**（57122419 出分後的下一個名額）當且僅當：整合通過，且 **keys30 至少 1 個驗證通過且正確**（證明增強→驗證→反變換→合併整條路徑；反變換另有 120 題×6 增強的單元測試），且精確度沒有災難＝（驗證通過數 <3）或（正確／通過 ≥0.5）。0 個驗證通過 → 不送，先檢討上限／併發。
- 整合失敗 → 不送；修正後下週（10/17 起 20k／週）再跑一次 dev run。
- 送前仍要：API 獨立審查（含改進建議區）、隱藏資料形狀測試（`runs/arc-agi-2/v13/tests/test_v13.py` 的 rerun hidden-shape／skip／abort 測試）。

## 下檔與對照（API 審查 #6）
- 比較對象是 v11（含 pass B）：keys30 上 v13 對 F（v8 全成本）+1.17／30，**在 F 的種子雜訊內，尚未證實**；對 H（v11）約 +1.4。
- 下檔：若 rerun 時 LLM 階段被跳過或沒時間，v13＝v11 減去 pass B（keys30 上 H−HA≈+0.6／30，雜訊大）→ 約 −0.5pp 期望損失。

## 離線估計（最終選定用；`artifacts/verification/arc-agi-2/v13-offline-v1.json`）
- keys30（30 題 task-score，E1 三種子）：HA 6.78（6.67／7.0／6.67）→ **v13（q38 程式）8.78**（8.67／9.0／8.67）；F（v8）7.61（9.0／5.83／8.0）。v13 對 F 逐種子差 −0.33／+3.17／+0.67（均 +1.17，在 F 的種子 SD≈1.6 內）；對 HA +2.0（SD 0.19），全部來自 F／H 三種子都 0 分的兩題 → 是互補，不是雜訊。
- 注意：P0b 只完成 26/60 個 round-1 job（覆蓋是下界）；keys30 是公開 eval，Qwen3.8 汙染無法排除（P0b 有增強）；加入 q36a3b 反而讓種子 2 從 9.0 掉到 8.67（驗證通過但錯的程式會擠掉 NVARC top-2）→ 只用一個寫手模型。
- 未量到：長上下文（>20k）的實際吞吐（P1 的 463–517 tok/s 是 720 秒短窗、每條約 1 萬 token）；64k 上限在 4×L4 conc 32 時每個工作約需 60–100 分鐘，rerun 才會量到。

## 修正（2026-10-13 17:10Z，v1 dev run 之後；判定規則不變）
- v1（`arc-2026-v13-progsynth-v1`）結果：整合全部通過；cap 20k 下 66 個工作 0 個自然結束、推理草稿有程式的只有 1/66 → 驗證 0 → 依本檔規則**不送 v1**。
- 修正 run＝**v2 `arc-2026-v13-progsynth-v2`**：commit 上限改為與 rerun 相同的 **64,000**（原預先寫的 40k 改掉，理由：P0b round-1 token 中位數就是 64k、平均 62k，40k 很可能重現 v1 的空結果；目的是讓 commit 與 rerun 只差任務集與牆鐘）；`COMMIT_SECONDS` 7,200（規劃扣額 (7,200+600)×2.2＝17,160；v1 實際只用規劃的 0.77）；併發公式 0.5·cap → **0.35·cap**（v1 KV 最高 0.64；64k 時約 13 條）；每列加記 reasoning_len／content_len，未自然結束者記推理最後 2k 字；移除無效的 `VLLM_FORCE_ATTN_BACKEND`（vLLM 0.19 標為未知變數；`--attention-backend TRITON_ATTN` 才是生效的那個）。
- 判定規則照舊：整合通過＋keys30 ≥1 個驗證通過且正確＋精確度下限。檢定力估計：64k、約 13 條、每條約 28 tok/s → 每個工作約 38 分鐘、84 分鐘生成窗約 25–30 個工作；依 P0b 19% 自然結束率 → 約 3–5 個結束 → P(≥1 驗證正確) 約 0.9。
- 本週 ARC 只剩約 10k → v2 在 10/17 00Z（新週 20k）之後才推送。
- 不放進 v2（留作 v3 候選）：預算強制（截斷時補 `</think>` 並強制寫程式碼的續寫）。最大的方法槓桿（64k 時 81% 截斷），但提示建構路徑未測，且 P1 log 顯示 prefix caching 關閉 → 每次續寫要重新 prefill 約 7 萬 token。

## 時程修正（2026-10-14 00:3xZ，推送前寫定；方法與判定規則不變）
- v2 推送由 10/17 00Z 提前到 10/14：協調者 amendment_2026_10_14（借額 until_2026-10-16，ARC 60,000）讓規劃扣額 17,160 通過滾動 7 日佇列。build（`upload-arc-2026-v13-progsynth-v2`，notebook f93be126…）、旗標、cap 64k、COMMIT_SECONDS 7,200、判定規則全部照上一段，不做任何修改。
- 另：先前「10/17 起 ARC 20,000／週」是過時資訊（amendment_2026_10_09 已撤回、amendment_2026_10_12 未啟動）；10/17–11/02 的分配由協調者 10/16 15:43Z 定。

## v3 預登記（2026-10-14 03:05Z，依 v2 dev run 數據寫定；v3 dev run 推送前不改）
- **唯一改動＝預算強制**（`runs/arc-agi-2/v13/src3/`，`--force-at 58000 --answer-tokens 6000`，cap 仍 64,000、併發公式、COMMIT_SECONDS 7,200、任務集、排序全部同 v2）。不併入其他機制（審查建議的「串流中途驗證草稿提早結束」留作 v4；不開 prefix caching——此混合模型預設關、未測）。
- **B／R 怎麼選**：v2 兩個自然結束在 47,069 與 56,074 token，且都驗證正確 → B 必須晚於 56,074（不能拿確定的贏換猜測）→ **B＝58,000**；R＝64,000−58,000＝6,000（強制尾段直接開 ```python，ARC 程式通常 1–3k token），總長不變 → 每工作仍約 55–60 分鐘、每 LLM 小時約 13 個工作（rerun 每小時 1 波）。續寫要重 prefill 約 75k token（prefix caching 關），v1 負載下約 45 秒／次。
- **同種子配對**：第一個請求的 seed 與增強 salt 由 (task, rep) 決定，與 v2 相同 → 前 58k token 近似重現 v2（vLLM 批次／MTP 數值不保證逐位相同），v3−v2 差異主要來自強制。
- **預期**：v2 第一波 13 個完成工作中 11 個撞上限；尾段約 5 個已在寫／驗程式、6 個還在找規則 → 預期強制觸發約 10 次、產出可驗證程式約 3–5 個、驗證通過 1–3、正確 0–2 → 相對 v2 預期 +0.5～+1.5 題／30（雜訊大）。
- **v3 取代 v2 成為下一筆提交，當且僅當全部成立**：
  1. 整合 5 項照舊，且 `token_id_parity.ok` 為真、`force_disabled` 不存在；
  2. 強制實際執行（`forced` 列無 `force_error`）≥3 次；
  3. 全體精確度下限照舊（驗證 <3 或 正確／驗證 ≥0.5），**且 forced 來源另設同一下限**（forced 驗證 <3 或 forced 正確／forced 驗證 ≥0.5）；
  4. 用 `replay_v2_merge.py` 同法把 v3 程式合併到 E1 HA 三種子，keys30 均值 **≥ v2 的 8.778**（v2 重播值）。
- 不成立 → v3 不送，v2 維持為現行提交；記錄原因，v4 再議。
- 送前照舊：hidden-shape（`tests3`）、API 獨立審查（含改進建議）、audit `submission_allowed`。

## v3 預登記修正（2026-10-15 04:2xZ，推送前；依 API 推送前審查 `artifacts/verification/arc-agi-2/api-reviews/20261015T035039Z-pre-launch-review-v13-v3-dev-run-budget-forcing.md` GO-WITH-FIXES；只加嚴、不放寬）
- **程式修正（`src3/v13_llm.py` → v13-llm-v3b）**：F1 續寫 http_error／exception／無文字／未完成都記 `force_error`（不再算成功強制）；**執行期斷路器**：前 3 次強制全失敗 → 之後開始的工作照 v2（cap 64k、不強制），記 `force_disabled`；F2 parity 改走真實路徑——`</think>` 必須是單一 token 且在 FORCE_TEXT 中恰出現一次（suffix 只在啟動時 tokenize 一次並快取），一個開思考、同模板參數與取樣、被 max_tokens 截斷的請求須 ids 數＝usage，再對其 ids（＋suffix）做一次 `/v1/completions`（32 token）須回 200 且有文字，任何失敗 → 整場關強制（v2 行為）；是否已離開思考改以 output ids 是否含 think-end id 判斷；強制的時間守門由「中止前 60 s」改為 `stop_new_at`（commit 中止前 5 分、rerun 15 分）；續寫記 `cont_t_first`（排隊＋prefill 時間）。tests3 新增 answer 模式、續寫被拒→force_error＋斷路器、parity 續寫失敗→關強制。
- **送出規則修正**：第 2 項改為「≥3 個強制列的續寫真的跑完（`cont_state == done`、`cont_completion_tokens > 0`、無 `force_error`）」；**新增第 5 項**：「≥1 題由 forced 來源驗證且全對、且該題在 HA 重播的平均增量 > 0（強制確有貢獻）」；斷路器觸發即第 1 項不成立。`audit_v13.py` 已照此實作（`v3_forced_executed_ge3`、`v3_forced_contributed`）。
- **報告（不設門檻）**：同一 run 的配對差＝`replay_v2_merge.py … --exclude-source forced` 對比完整重播；`cont_t_first` 分布與 vLLM 搶占計數。第 4 項（HA 重播均值 ≥ v2 8.778）只當「不退步」門檻：v2 rep 0 只 9/30 題跑完、且 temperature 1.0 下 v3 前 58k 不會重現 v2 軌跡（兩者是近乎獨立的抽樣），虛無假設下通過機率約 0.6 → **不得以第 4 項宣稱 v3 有增益**。
- 已知未量到的風險（記錄，不擋）：58k–64k 之間的自然結束／首次驗證草稿被強制取代；第一波約同時到 58k 造成的 prefill 尖峰（rerun 最後一波可能滑過 `abort_at`）；隱藏集上 forced 偽陽性（keys30 為公開資料，精確度偏樂觀）。

## v3 送出規則修正 2（2026-10-16，T3 審查後、推送前；只加嚴）
審查：`artifacts/verification/arc-agi-2/reviews/20261016-v3-e4-chain-summary.md`（Opus 5.5＋Argon 平行盲審＋advisor 裁決）。
- 第 4 項的 v2 基準檔 `replay-v2-merge.json` 釘住 sha256（3676258e…）；雜湊不符則第 4 項不成立。
- 第 5 項「強制有貢獻」改用**配對差**：某題 forced 來源驗證且全對，且該題在完整重播的平均增量 − `--exclude-source forced` 重播的平均增量 > 0（舊實作用完整重播的增量，會把同題自然結束的程式也算成強制的貢獻）。
- 新增第 6 項：完整重播均值 ≥ 排除強制的重播均值（強制不得淨負）；第 7 項：`lost_correct_nvarc_any` 為 false（同 v2 的 F2）。
- 已知小偏差：每個工作列只記第一個通過的候選（content→forced→reasoning），排除重播會連帶拿掉被 forced 搶先的思考草稿 → 配對差略偏向高估強制的貢獻。
- 提交前審查 JSON：至少兩份原始審查（檔案＋sha256＋各自判定）、沒有 NO-GO、每條 P0／P1 為 REFUTED 或 FIXED_OUTSIDE_KERNEL（附證據），且 `kernel_code_change_required` 為 false；任何需要改已推送 kernel 的 P0／P1 → 不送。
