# 預登記：Boyko V22 fork（arc-2026-fork-boyko-v1）作為第 2 名額候選 —— 2026-11-01 12:0xZ，推送前、無任何輸出

## 動機
- 使用者 11-01「讓arc全力推進」（協調者轉）；GPU 本週剩 7,577 s 全撥 ARC，否則過期。
- 現第 2 名額 v8 56864814 公開 30.97（同碼 3 次 30.97／29.72／29.72，均 30.14）＜ 現銅牌線約 32.22 → R4「公開 ≥ 銅牌線」條件不符。

## 是什麼（誠實定位）
- 來源 `evgendvorkin/arc-2-2026-4-4-local-boyko-nvarc` V22（sha256 3869a268…，10-31 拉取）。權重 `konstantinboyko/qwen3-4b-bfloat16-v02-01-01` 經 10-31 查證＝sft139 重傳。
- 架構＝NVARC pass A（seed 42/1/2）＋換種子 pass B（137/17/29）池化、保留 pass-A 第一名 ＝ **與 v8 同一設計**，差在作者實作（cheap-first 排序、catch-up pass、PYTHONHASHSEED=0、950 s 單題上限等）。**不是新家族**。
- 唯二改動（`runs/arc-agi-2/fork-boyko/build_fork.py`）：(1) commit 模式預算 75→25 分（rerun 分支 12h−15m 不變）；(2) `ARC_ASTRA_PROMOTE` 預設 '1'→'0'（關閉 V21/V22 新增、作者自述未驗證的 D4 專家提升；等於 V17–V20 的純神經 A+B 選擇）。
- **警告**：作者表格的公開分 31.81–32.64（V17–V20，均 32.12；作者隊榜 32.22）不是由 V22 碼產生，且連結指向另一本 notebook（arc-2-2026-3-4）；我們從未量過此碼。32.12 不是我們的估計。

## 決策規則（看到任何輸出前寫定）
1. 送出閘門（R1）：隱藏形狀測試（`tests/test_arc_agi_2_fork_boyko_hidden_shape.py`）、kaggle_env_lint（K1/K2 RISK 已讀：路徑與作者已執行版相同、-U 只在離線 wheelhouse）、commit run audit（4 題 smoke 有真答案、無 Traceback、finalizer 完成、submission 全 id）。外加 T3（Argon＋Opus API＋advisor），在等出分期間做。
2. 出分後：公開分 **≥ 出分當下的銅牌線**（讀不到則用 32.22）→ 提議第 2 名額由 v8 換成此 fork；**< 銅牌線 → 不變**。不比較與 v8 的差是否超出雜訊帶（R4 只要求 ≥ 銅牌線且最不相似；此 fork 與第 1 名額的相似度不高於 v8：同 NVARC、無 LLM 通道）。
3. 換的流程：問使用者一句＋T3＋Fable（advisor）＋回報協調者；**Kaggle UI 勾選由使用者本人做**。
4. 時限：**11-02 23:00Z 前未出分 → 不換**（留時間給使用者點選）；不另送其他提交（只有一筆評分名額，v5b 不送）。
5. 第 1 名額不變（57165541）。

## 補充 1（11-01 12:2xZ；推送後、無輸出）
- T3 Opus P2-5 指出門檻規則受雜訊主導（單次抽樣過 32.22 約 45%；31.8 仍比 v8 均值高 ~2.5 sd 卻會被拒）。**維持原規則（≥ 銅牌線）**：R4 字面要求、協調者 11-01 確認；代價記錄於此，不在看到分數後改。

## 補充 2（11-01 12:18Z；kernel 仍 QUEUED、無任何輸出）—— commit 預算造成的不完整
- 25 分預算含安裝，4 題 smoke 可能被 end_time 截斷；GPU 不足以重跑。
- 若**唯一**失敗的 audit 閘門是 `smoke_tasks_finished` 及／或 `selector_saw_passA_outputs`，且 `no_crash_or_traceback`、`passA_rc0`、`finalizer_ok`、`submission_full_coverage_valid`、`four_workers_started`、`promote_off`、`upload_matches_pinned_build`、`hidden_shape_test_passed`、`independent_review_go` 全過，且 ≥2 個 smoke 題有 `finished` 行、selector pass-A outputs ≥2 → 視為 PASS-with-note 並送出（audit JSON 另記 `prereg_addendum_2_applied`）。其他任何閘門失敗 → 不送、不重推。
- 已知殘餘風險（T3 O-P2-1）：隱藏重跑尾端（在途 TTT＋expert 共用 15 分緩衝）可能使評分 ERROR；ERROR 不影響現有選定 v2＋v8，不視為工具失敗。

## 使用者同意（11-01 12:27Z，在 ARC 對話中，出分前）
- 使用者原話：「可以，過銅牌線就換，我會自己勾選」。→ 規則 3 的「問使用者一句」已滿足（僅限本 fork、公開 ≥ 出分當下銅牌線）。Kaggle UI 勾選由使用者本人做；出分後仍照走 Fable（advisor）確認＋回報協調者。

## 使用者同意代勾（2026-11-02T01:11Z，ARC 對話）
問：57871700 出分後若 ≥ 當時銅牌線且 11-02 23:00Z 前出分，要不要由 ARC 代勾（只在 Kaggle Submissions 頁取消 56864814、勾 57871700、保留 57165541，讀回 2/2）？使用者原話：「可以」。規則本身（門檻、期限）不變。
