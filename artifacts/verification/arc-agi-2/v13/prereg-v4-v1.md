# v4 預登記（2026-10-16，E4／v3 出結果前寫定；推送前另需 T3 審查，見文末）

依據：`prereg-e4-cap96k-v1.md` 第 4 項（「v4 的建置、hidden-shape、推送前審查與 ship rule 另寫預登記修正」）。本檔就是那份修正，在 E4 與 v3 的結果出來**之前**寫定，結果出來後只照表選，不改規則。

## 1. 四個預先建好的變體（全用 `build.py --src src3`，build.py／src3 與 v3 相同、未改）
| slug | llm-args | 何時用 |
|---|---|---|
| `arc-2026-v13-progsynth-v4-c80` | `--cap 80000` | E4＝80000 且 v3 **未**通過自己的預登記；或 v3 通過但 E4 有 >74k 的驗證結束（見 c80f） |
| `arc-2026-v13-progsynth-v4-c96` | `--cap 96000` | E4＝96000（不論 v3 是否通過；見 c96f） |
| `arc-2026-v13-progsynth-v4-c80f` | `--cap 80000 --force-at 74000 --answer-tokens 6000` | E4＝80000 且 v3 通過，且 E4 沒有任何驗證通過的自然結束超過 74,000 token（否則用 c80，T3 Opus P2-5：74k 強制會切掉 74k–80k 之間的自然結束） |
| `arc-2026-v13-progsynth-v4-c96f` | `--cap 96000 --force-at 90000 --answer-tokens 6000` | **永不推**（T3 Opus P2-6：dev run 到不了 90k，強制在推送前零次實測）→ E4＝96000 且 v3 通過時也用 c96 |
- E4＝64000 或 inconclusive → **不推任何 v4**（四個資料夾都不推，記錄「建好未推」）。
- **E4 守門 3 要求改併發**（KV 峰值 ≥0.95 或有 preemption → 依 E4 推送前修正 2／3 重算併發）→ **不推預建的 v4**（預建版都用 KV 公式的 11／10，build.py 也不收 `--conc`）；要推就得另建帶 `--conc` 的版本並重新審查（T3 Opus P1-2／Argon P0-2）。
- 「v3 通過」＝ HANDOFF 記錄的 v3 判定為通過預登記規則（prereg-dev-run-v1 v3 各項＋送出規則修正 2）；與 v3 是否已送出、是否已出分無關。
- 四個 notebook 之間只允許 v13_llm.py 呼叫那一格與 `preparation.json` 不同（建置後 diff 存證）；與 v3 notebook 相比也只允許那一格不同。`bare` slug `arc-2026-v13-progsynth-v4` 不使用（建置不可覆寫）。
- 併發照 v13_llm.py 的 KV 公式自動決定（KV 331,200 → 80k：11、96k：10），不另傳 `--conc`。

## 2. dev（commit）run 量得到什麼——先寫明，避免事後解讀
- build.py 的 COMMIT_SECONDS＝7,200（發射路徑綁定，不改）。v2 dev run 實測：生成從 648 s 開始、5,512 s 中止（abort＝end−6 min，停派＝abort−5 min），第一波 13 個工作做完、第二波 13 個全被中止。
- 每工作約 18 tok/s（E3；10 併發時可能到約 22.6）。80k：一個工作 3,540–4,440 s → **第一波 11 個大致做得完**，第二波被中止。96k：4,250–5,330 s → **第一波 10 個可能做不完**（18 tok/s 時會在約 87k token 被中止，只剩思考草稿可驗證）。強制：80k 的 74k 強制點在 18 tok/s 時約 4,760 s 到達，做得完；96k 的 90k 強制點在 dev run 內**到不了**。
- **以 keys30 題數計**（T3 Opus P1/P2-4）：commit 的工作順序是 4 個 NVARC smoke 題在前、再 keys30（`job_order`）→ v2 第一波 13 個工作只有 9 個 keys30 題。80k：第一波 11 個工作 ≈ **7 個 keys30 題**，且要每工作 ≥16.4 tok/s 才做得完（74k 強制點要 ≥16.2 tok/s 才在停派前到達）；96k：≈6 個 keys30 題、可能 0。64k 以上的解碼速度沒量過，可能低於 18 → 以上偏樂觀。
- 結論：v4 的 dev run **不是**上限效果的量測（那是 E4 的工作），只是整合測試＋已完成工作對 NVARC 的「不傷害」檢查（約 6–7 題）。v4 的離線證據＝E4 判定（分層模型）＋這裡的不退步檢查；依 `final_selection_amendment_2026_10_13`，只有模型估計、沒有多次離線均值的 v4 **只能當第 2 名額候選**，除非之後另有多次離線重複。

## 3. v4 送出規則（全部成立才送；任何一項不成立＝不送、記錄）
1. 整合閘門：`audit_v13.py audit <slug>` 的 integration 各項通過（同 v2／v3：完成、submission 形狀、無例外、sandbox、parity 若有強制）。
2. `v4_cap_applied`＝true（v13-llm.json 的 cap＝變體的上限，且每列 cap ≤ 上限）。這個 gate 從沒對真的非 64k 輸出跑過 → E4 下載後先對 E4 輸出呼叫 `audit_v13.v4_cap_applied(llm, rows, 96000)`（patch 後的獨立函式；audit 本身拒絕 E4）應為 true；若 gate 本身有錯，修 audit（不動 kernel）後才審 v4。
3. hidden-shape：`audit_v13.py hidden-shape <slug>` 通過，hash 與推送版本相符。
4. **對 NVARC 不傷害**（不是對 v2 不退步；T3 Opus P1-3／Argon P2-8）：重播輸入＝原始 `runs/arc-agi-2/v6/output-<slug>/v13-programs.jsonl`（同 v3，含中止列的思考草稿），`replay_v2_merge.py runs/arc-agi-2/gcp-e1/out/e1 <該檔> artifacts/verification/arc-agi-2/v13/replay-<slug>-full.json`；**每個 HA 種子 delta ≥ 0 且 `lost_correct_nvarc_any`＝false**（gate `v4_replay_no_harm_vs_ha`／`v4_no_lost_correct_nvarc`，檔名綁定 slug）；重播均值 − v2 的 8.778 只記錄。c80f 另做 `--exclude-source forced` → `replay-<slug>-excl-forced.json`，並要：強制點＝74,000 且 parity 通過、未停用（`v4_forcing_active`）、至少 1 次強制真的跑完（`v4_forcing_executed`；delta 審查 P2-1）、forced 來源精確度下限（`v4_forced_precision_floor`）、完整 ≥ 排除強制（`v4_forcing_not_negative`）（T3 Argon P1-6）。永不寫 replay-v2-merge.json。
5. 提交前審查 JSON（同 v3 送出規則修正 2 的格式：≥2 份原始審查＋sha256、無 NO-GO、每條 P0／P1 為 REFUTED 或 FIXED_OUTSIDE_KERNEL、`kernel_code_change_required`＝false）。
6. 同時只能有 1 筆在評分：v3（若已送）出分或失敗前不送 v4；`cloud.py submit` 會即時再查。

## 4. 推送順序（有人值守；E4 判定出來後）
1. 照第 1 節的表選 slug；HANDOFF 記錄 E4 判定、v3 判定、選到的 slug。
2. **E4 的發射紀錄 `artifacts/verification/arc-agi-2/v6/launch-arc-2026-v13-e4-cap96k-v1.json` 存在後**（或 03:37Z 備援已記錄「E4 不推」且之後沒有排定的 ARC 發射）才套用 `v13/patches/audit-v13-v4-gates-v1.patch`（與 launch 的 gate_e5 patch）並跑 `tests_e5/test_e5.py`（MODULE_SUFFIX=）＋`tests_e4/test_audit_v3_gates.py`；`run_tiered_review.py check <T3 修正後 delta 審查的 manifest> --tier T3 --require <本檔、patch、選到的 upload 資料夾檔案>` → exit 0 才繼續（雜湊不符 → 只把 diff 送一家 Argon，無 P0 才繼續）。
3. 資源：10/17 週 ARC 已扣＋17,160（2.2 ×（7,200＋600））≤ ARC 份額（04:40Z 後為 67,140）。沒有其他 ARC kernel 在跑（E4 已 terminal）。
4. `~/.local/share/uv/tools/kaggle/bin/python runs/arc-agi-2/v13/launch_v13.py --gate-only runs/arc-agi-2/v13/upload-<slug>` → 同指令去掉 `--gate-only`（任何錯誤＝停，不重試未知結果）。
5. terminal 後：status → settle → robust_download → audit（第 3 節）→ 提交前審查 → 第 3 節全部成立且無在評分的提交才 `cloud.py submit`。

## 5. 審查（T3 已審：`reviews/20261016T134356Z-v4-e5-prep-{opus,argon}.md`，兩家 GO-WITH-FIXES；逐條處理見 `reviews/20261016-v4-e5-T3-summary.md`）
- T3（累計 ≥15,000 GPU s、通往正式提交）：Argon（`agy_ask.py --argon`）＋Opus 5.5 API 平行盲審＋advisor 裁決；本檔、四個 upload 資料夾的 notebook diff、build.py 相關段落、audit_v13 的 v4 gate、E5 預登記一併送審（一份 manifest），推送時用 `check --require` 綁定。
