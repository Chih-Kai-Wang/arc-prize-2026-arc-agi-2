# fork-perfpatch：mikelou1/arc-agi2-lb33-89-minimal-perfpatch 的 AS-IS fork

- 來源：公開 notebook `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`，sha256 `ee49384a10a4296dc0bf0670ae67325521f798d5337ed57bee50c8a443a378f4`（來源副本：`runs/arc-agi-2/v6/external-src/mikelou1__arc-agi2-lb33-89-minimal-perfpatch/`）。
- `upload-arc-2026-fork-perfpatch-v1/arc-2026-fork-perfpatch-v1.ipynb` 與來源逐位元組相同（零修改）；只有 `kernel-metadata.json` 是我們的（私有、離線、4xL4、v6 同一組 model/kernel/docker sources；來源 metadata 的 model 與 kernel source 已確認一致）。
- `src/` = 所有 `%%writefile` cell 本體（`cloud.writefile_cells()` 抽出）：`arc_loader.py`、`arc_decoder.py`、`arc_solver.py`、`starter.py`。notebook 的非 writefile cell：標題 markdown、`global_end_time = time.time() + 12*3600 - 600`、`!pip uninstall -y tensorflow`、啟動 `starter.py`、最後的組裝 cell。
- 檢查：`cloud.local_gate()` 通過；`tests/test_arc_agi_2_fork_hidden_shape.py`（10 項）；`audit_fork.py` + `tests/test_arc_agi_2_fork_audit.py`（11 項）。

## commit 模式 vs `KAGGLE_IS_COMPETITION_RERUN`（以 `os.getenv` 是否為真判斷，starter / arc_solver / 組裝 cell 各自判斷）
| | commit（無該環境變數） | rerun（有） |
|---|---|---|
| 資料檔 | `.../arc-agi_evaluation_challenges.json`（120 題） | `.../arc-agi_test_challenges.json`（動態讀取，公開檔 240 題） |
| 入列任務 | **寫死 4 題** `0934a4d8, 36a08778, 981571dc, aa4ec2a5`（`sorted(keys)` 後過濾） | `sorted(data.keys())` 全部，再放 4 個 `None` 結束標記 |
| worker | 4 個 rank（`mp.spawn nprocs=4`），每題：128 條增強做 rank-256 rsLoRA 測試時訓練 → 16 視角 turbo-DFS 解碼 → 增強重評分；每題解碼上限 1200 秒、全域到 `global_end_time` | 同左，全部任務；逾時未處理的題留預設答案 |
| 組裝 cell | 讀 evaluation challenges + solutions，組完後 `benchmark_selection_algos()` 並印 `*** Reload score: X`（`validate_submission`） | 只讀 test challenges，不評分 |

## submission.json 怎麼來
最後一個 cell（無 try/except）：`ArcDataset.from_file(challenges)` →（commit 才 `load_replies`）→ `ArcDecoder(data.split_multi_replies(), n_guesses=2)` → `decoder.load_decoded_results("/kaggle/inference_outputs")`（worker 寫的 bz2 pickle，檔名 = 子鍵 `<id>_<i>.<增強>`，內容是 `beam_score / score_aug / solution` 的 list）→ `data.get_submission(decoder.run_selection_algo())` → `json.dump` 到 `submission.json`。`run_selection_algo()` 預設用 `score_kgmon`；`get_submission` 先對 **每個** challenge id、每個 test input 填 `{"attempt_1": [[0]], "attempt_2": [[0]]}`，再以 `fill_submission` 用前 2 個選出的解覆蓋。所以沒解出的題仍有合法預設值，id 集合由 challenges 檔決定。因此 commit 模式的 submission.json 涵蓋全部 120 個 evaluation id（只有 4 題真的跑過），audit 以此為預期 id 集合。

## 是否動態讀 test 檔
是。rerun 模式沒有任何寫死的 id 或題數：starter 與 arc_solver 都是 `json.load` 後 `sorted(keys)`；組裝 cell 同樣讀檔。四個寫死 id 只出現在 `starter.py` 的 `if not rerun_mode:` 分支（測試已斷言）。

## 可疑點／風險（皆為讀碼所得，未執行 notebook）
- 無網路呼叫：src 內沒有 http/requests/urllib/socket/subprocess；模型以 `local_files_only=True` 從 `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1` 載入；`!pip uninstall -y tensorflow` 是本機移除、不下載。notebook 沒有外部下載。
- 沒有 v6 那種防護：沒有事先寫好的備援 submission.json、沒有單題例外隔離（見下節「starter 如何啟動」）。若 `/kaggle/inference_outputs` 不存在（所有 worker 都在 `makedirs` 前死掉），組裝 cell 會 `FileNotFoundError`（測試 `test_rerun_missing_outputs_dir_raises` 記錄此行為）。
- 沒有「task failed」之類的 log 字串；audit 以 `Traceback (most recent call last)` 與 `*** Reload score:`（組裝完成）取代。可用的 log 字串：`[Rank N] start!`、`finished <id> in Ns`、`timeout after Ns for puzzle <id>`、`allocated NMB`。
- 增強重評分的 seed 用 `hash(bk) % 1024**2`，受 PYTHONHASHSEED 影響，跨 process 不可重現（對正確性無影響）。
- commit 模式 `planned_commit_wall_seconds=2700` 是我們的計畫值；notebook 本身無 commit 的牆鐘上限，只有每題 1200 秒解碼上限與 12h-10min 全域截止。

## starter.py 如何啟動、worker 當掉的後果
notebook 倒數第二個 cell 是 IPython shell escape（子行程，不是 `%run`、也不是 `subprocess`）：
`!UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 python starter.py --end-time {global_end_time}`
（`{global_end_time}` 由 IPython 以第一個 code cell 的變數插值）。含意：
- `!` 命令的非零結束碼不會讓 cell 失敗（ipykernel 只回傳結束狀態、不丟例外），所以 starter 當掉後 notebook 會**繼續執行最後的組裝 cell**，不會整本失敗。
- `mp.spawn(join=True)` 在任一 worker 例外時會終止其餘 worker 並讓 starter 以非零離開；佇列裡未處理的題不會再被處理。結果是「部分輸出」：已寫出的解碼檔照常被讀，沒跑的題保持預設 `[[0]]`，submission.json 仍會產生（任務數少、分數低，但 id/格式合法）。
- 唯一會讓組裝 cell 失敗的是 `/kaggle/inference_outputs` 根本不存在（所有 worker 都在 `makedirs` 前就死）。
- 因此 audit 不能只看 submission 格式：另以 smoke 4 題必須有非 `[[0]]` 的答案、`[Rank N] start!`×4、4 題 `finished`、無 Traceback、Reload score > 0 把關。
