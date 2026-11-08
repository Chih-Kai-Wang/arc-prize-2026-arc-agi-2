# E5 預登記（2026-10-16，v3／E4 出結果前寫定；T3 審查後修正版）：v3 強制的同 run 配對差再加一波

## 為什麼
- 協調者 10-16：10/17 週 ARC 份額 52,140 → 67,140（+15,000，10/17 04:40Z 寫進 config），用在「提高私榜估計精度的離線驗證」。
- 隱藏集同 kernel 重跑是確定性的（57165541／57214357 都是 32.22）→ 重跑**估不出**雜訊；能提高精度的是 keys30 上的離線量測。
- v3＝v2＋58k 預算強制（src3 相同、只差旗標）。所以 v3 相對 v2 的增益＝**同一個 run 裡**「完整重播 − 排除強制來源的重播」（`--exclude-source forced`），兩者軌跡完全相同，只差強制答案 → 這是最乾淨的配對量測（T3 Argon 建議 3）。
- 不用「同種子＝接續被中止的工作」：temperature 1.0 下 vLLM 批次會讓長生成發散，v3 的前 58k 不會重現 v2（`prereg-dev-run-v1.md` 第 73 行、`prereg-e4-cap96k-v1.md` 限制段；T3 Opus P1-1／Argon P1-3 指正原稿的錯誤前提）→ E5 的工作是該題的**新抽樣**，不是續跑。
- v3 的 dev run 只有 7,200 s：約 9 個 keys30 題做完（v2 實測），強制配對差只有這幾題 → E5 在 dev run 沒做完的題上再跑**一波**，讓強制配對差的題數加倍。

## 設計（`runs/arc-agi-2/v13/build_e5.py`，新檔）
- 不跑 NVARC；src3 的 LLM 檔＋`v13_llm.py --jobs e5-jobs.json --force-at 58000 --answer-tokens 6000`（＝v3 建置的 llm args，從 v3 的 preparation.json 讀、gate 重算）；只在 commit 跑（rerun 會 assert 拒跑）、永不提交。
- **base 只能是 `arc-2026-v13-progsynth-v3`**（build 與 launch gate 都有白名單；原稿的分支 B 刪除——96k／80k 的一波在 6,200 s 內做不完或覆蓋不到 15 題，T3 Opus P2-7／Argon P1-4）。
- 工作＝keys30 中 v3 dev run 的 rep-0 列**沒真正做完**的題（中止、或到 58k 但強制被跳過／續寫中止——`build_e5.finished()`，T3 Argon P0-1），依 keys30 順序在前，再接沒開始的題；**最多 13 個＝一波**（v3 的 KV 併發 13；T3 Argon P1-5：第二波一定被中止，白燒約 3,300 GPU s）。全是 rep 0。
- E5_SECONDS＝5,200 → 規劃扣額 2.2 ×（5,200＋600）＝**12,760**。一波：設定約 700 s＋58k 在 ≥17.2 tok/s（v2 第一波最慢）＝4,070 s < 停派 4,540 s（強制仍會開始）＋續寫 prefill 與 ≤6k 答案約 400 s ≈ 4,470 s < 中止 4,840 s。
- 10/17 週 ARC 合計規劃：v3 17,160＋E4 17,820＋v4 17,160＋E5 12,760＝**64,900** ≤ 67,140（餘 2,240，可容一次小重試；用不到的在 10/22 12:00Z 前釋出）。

## 何時跑（分支，寫定）
- **A**：v3 通過自己的預登記（HANDOFF 記錄的 v3 判定）→ 跑 E5，slug `arc-2026-v13-e5-v3wave-v1`。
- **C**：v3 未通過 → 不跑 E5；15,000 在 **10/22 12:00Z 前釋出**（寫進 RESOURCE PLAN）。

## 推送前提（有人值守）
1. 10/17 04:40Z 之後（config 已寫 67,140；check2 才看得到）。
2. v3 dev run 已 terminal、settle、下載；沒有其他 ARC kernel 在跑（E4 與 v4 dev run 都 terminal）。
3. **E4 的發射紀錄 `artifacts/verification/arc-agi-2/v6/launch-arc-2026-v13-e4-cap96k-v1.json` 存在後**（或 03:37Z 備援已記錄「E4 不推」且之後沒有排定的 ARC 發射）才套用 `v13/patches/launch-v13-gate-e5-v1.patch`（launch_v13.py 綁在 00:07Z／03:37Z 發射路徑上；E4 的備援推送也要用未改的版本），`tests_e5/test_e5.py`（MODULE_SUFFIX=）通過。
4. 10/17 週 ARC 已扣＋12,760 ≤ ARC 份額。
5. `run_tiered_review.py check <T3 修正後 delta 審查的 manifest> --tier T3 --require <build_e5.py、patch、本檔>` exit 0；`launch_v13.py --gate-only` → 推送（任何錯誤＝停）。

## 主估計（預先寫定）
- 每個來源（v3 dev run、E5）各自做兩次重播：`replay_v2_merge.py runs/arc-agi-2/gcp-e1/out/e1 <該 run 的原始 v13-programs.jsonl> <dest>`，以及同指令加 `--exclude-source forced`；檔名 `replay-v3-full.json`／`replay-v3-excl-forced.json`（v3 cron 已產生）與 `replay-e5-v3wave-full.json`／`replay-e5-v3wave-excl-forced.json`。永不寫 replay-v2-merge.json。
- **強制配對差 F**（題／30，keys30 計分）：題集合分開——D＝v3 dev run 中 `finished()` 的 keys30 題，E＝E5 中 `finished()` 的 keys30 題（與 D 不重疊；F、F_E、n、bootstrap 都用同一個 D∪E；delta 審查 P2-2）。每個 HA 種子 s，F_s＝Σ_{t∈D}(per_task_delta_full − per_task_delta_excl)[v3 dev, s, t]＋Σ_{t∈E}(同)[E5, s, t]（dev run 未做完的列即使有部分強制答案也不計，避免同題算兩次）；F＝三種子平均。**題級 bootstrap SE**：對兩個 run 合併後有完成工作的題重抽（10,000 次，固定種子 0），SE＝F 的標準差。
- **覆蓋題數 n**＝兩個 run 中真正做完的 rep-0 keys30 題數（`finished()`）。
- 描述用（不進決策）：E5＋dev 的 v3 rep-0 每題分數 vs v2 三個 rep 的每題平均（未配對 Δ，同題集合）。

## 怎麼用（決策規則，寫定）
- v3 的離線估計＝v13 v2 的預登記調整估計 7.753＋**0.65 × F**（F 是在 keys30 上量的、強制只在被做到的工作上起作用，隱藏集覆蓋 c＝0.65；T3 Opus P1-1／Argon P1-3）。
- **0.65 是刻意保守的**（advisor 10-16）：F 只量了約 22 個有做完的題，隱藏集的期望其實約 0.65 × 30 × F／22 ≈ 0.89F；這裡故意不乘 30／22 的覆蓋係數，讓規則偏向維持 v13 v2。決策當下不得「修正」成 0.89。
- **第 1 名額換成 v3** 只在：n ≥ 15，且 0.65 × F ≥ max(0.5, 2 × 0.65 × SE)，且 **E5 自己那部分 F_E（三種子平均）> 0**（D 的題是 v3 通過「強制有貢獻」時用過的，有選擇偏誤；E 沒被挑過，當獨立確認；T3 Argon P1-3），且 v3 公開分不低於 32.22 − 2.4（公開雜訊帶，見 HANDOFF 候選表）。否則維持 v13 v2。
- 第 2 名額照舊是 NVARC 避險（v8）；v3 與 v13 v2 高度相關，不當第 2 名額。
- E5 不提交、不改任何 kernel。

## 審查
- T3（Argon＋Opus 5.5＋advisor），與 v4 預登記同一份封包：`reviews/20261016T134356Z-v4-e5-prep-{opus,argon}.md`，兩家 GO-WITH-FIXES；本版是修正後版本，修正內容另送 delta 審查（`reviews/20261016-v4-e5-T3-summary.md`）。
