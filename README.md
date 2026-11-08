# ARC Prize 2026 (Kaggle "ARC-AGI-2"): NVARC test-time training + Qwen3.8 program synthesis

[English](#english) | [繁體中文](#繁體中文)

---

<a id="english"></a>
## English

### Summary
This repository is our team's (Kaggle `f0909172434`) code and experiment records for the Kaggle competition **ARC Prize 2026 – ARC-AGI-2** (`arc-prize-2026-arc-agi-2`, closed 2026-11-02 23:59 UTC). Our best selected submission runs the public NVARC-style Qwen3-4B test-time-training pipeline (from sorokin's Apache-2.0 notebook, hardened by us) for one pass, then frees the four L4 GPUs and lets **Qwen3.8-27B-FP8** (served by vLLM) write Python `transform(grid)` programs; a program's answer is used only if it reproduces every augmented training pair. It scored **32.22** on the public leaderboard. Our second final selection is a two-pass variant of the NVARC pipeline (**30.97** public). Most later ideas were pre-registered, measured offline and rejected; they are documented here with their numbers. The project was run by AI coding agents under human supervision (see [Glossary](#glossary)).

### Results
Public leaderboard scores are the share of hidden test outputs solved (pass@2), on the public half of the hidden set. **Private leaderboard scores are not recorded in this repository.**

| Kaggle submission | Version | Public LB | Note |
|---|---|---:|---|
| **57165541** | **v13 v2**: NVARC pass A + Qwen3.8-27B-FP8 program synthesis | **32.22** | **Final selection 1** |
| **56864814** | **v8**: NVARC pass A + re-seeded pass B, pooled | **30.97** | **Final selection 2** (hedge: different seeds/family) |
| 57828532 | v13 v5: v2 code with fresh seeds, fixed concurrency 16 | 33.89 | same family as selection 1, +1.67 pp is inside our 2.4 pp noise band → not selected |
| 57214357 | v13 v2 resubmitted (same kernel version) | 32.22 | identical score: the rerun is near-deterministic |
| 57871700 | fork of evgendvorkin V22 ("Boyko") | 30.56 | below the bronze line at the time → not swapped in |
| 57001190 / 57064408 | v11 (half-cost TTT width) r1 / r2 | 30.56 / 28.89 | |
| 56894434 / 56935323 | v8 resubmissions r2 / r3 | 29.72 / 29.72 | v8 mean of 3 = 30.14 |
| 56917114 | v9 (confidence-ordered pass B) | 29.72 | |
| 56809468 / 56837247 | v6 (hardened author pipeline) / v7 (coverage scheduling) | 30.14 / 30.14 | |
| 57122419 | as-is fork of mikelou1 "perfpatch" | 30.14 | |
| 57301011 | v13 v4-c96 (96k-token cap) | 29.03 | −3.19 pp vs v2 |
| 56761143 | early pipeline: Qwen3-2B grid model | 4.58 | superseded (code not included) |
| 56751754 | early baseline: Qwen2.5-Coder-7B | 0.00 | superseded (code not included) |

- Bronze line read from the live public leaderboard on 2026-11-02 (about 18:40 UTC, 3,627 teams, rank 362): **32.22**. Our 32.22 was therefore on the bronze line on the public LB; our final public rank at the deadline is not recorded.
- Noise: same-family reruns gave a public-score SD of about 0.90 pp; we treated public differences ≤ 2.4 pp as noise and chose final submissions by pre-registered offline estimates, not by the public LB.
- Offline estimate (30 public-evaluation tasks "keys30", tasks solved out of 30, pre-registered grid of 2 reps × 3 seeds): v13 v2 raw 8.278, queue-order-adjusted **7.753**; v8 **7.61** (3 seeds). NVARC half-cost pass A alone ("HA"): 6.778.

Sources: `artifacts/verification/arc-agi-2/v13/` (pre-registrations, replays, slot-2 analysis), `artifacts/verification/arc-agi-2/v6/audit-*.json`, `artifacts/verification/arc-agi-2/v13-offline-v1.json`, `artifacts/verification/arc-agi-2/gcp-e1-result-v1.json`.

### Method
**Phase 1 — NVARC pass A (Qwen3-4B).** Code: `runs/arc-agi-2/v11/src/` (copied unchanged into `runs/arc-agi-2/v13/src/`). For each task: rank-256 rsLoRA test-time training of `sorokin/qwen3_4b_grids15_sft139` on augmented training examples (half-cost width: 8 train augmentations × 8 geometries = 64 sequences), 16-view turbo-DFS decoding, augmentation re-scoring, `kgmon` selection of the top two candidates. Our changes to the author's notebook (v6 onward): automatic path discovery; commit mode runs a 4-task smoke test and rerun mode runs every task; a complete fallback `submission.json` is written before any work; per-task exception isolation; independent worker processes per GPU; atomic writes; forced shutdown at the deadline. In v13 the second NVARC pass ("pass B" from v8) is switched off with an environment variable.

**Phase 2 — Qwen3.8-27B-FP8 program synthesis (`runs/arc-agi-2/v13/src/v13_llm.py`, `progsynth.py`).** After pass A is assembled into a valid submission, the GPUs are released and vLLM 0.19.0 serves Qwen3.8-27B-FP8 from a separate Python 3.12 virtualenv (tensor parallel 4, MTP speculative decoding with 2 tokens, fp8 KV cache). The model is asked to write `transform(grid)`; each candidate program runs in a sandboxed subprocess and is accepted only if it reproduces **all augmented training pairs**.

**Merge rule (pre-registered, `runs/arc-agi-2/v13/src/v13_merge.py`).** Per test output: the answer of the verified programs (majority; ties → first verified) becomes `attempt_1`, and NVARC's top-1 becomes `attempt_2`. If the program answer is already one of NVARC's two attempts, the entry is left unchanged; if NVARC has no real grid, `attempt_2` is the next distinct program answer. The submission is rewritten atomically after every verified program, and any failure in phase 2 leaves the NVARC submission in place.

**v8 (final selection 2).** Same NVARC pipeline (full author width), plus pass B: in the wall time pass A leaves unused, re-run tasks with independent seeds and pool pass-A and pass-B candidates for selection (`runs/arc-agi-2/v8/src/arc_v6_config.py`).

**Engineering practice.** Every submission was gated by hidden-shape tests (different task IDs, counts and order than the public file, injected worker crashes/hangs; e.g. `tests/test_arc_agi_2_v6_hidden_shape.py`), a commit-run audit (`runs/arc-agi-2/v13/audit_v13.py`, `runs/arc-agi-2/v6/audit.py`), and for larger GPU spends a written pre-registration with decision rules fixed before the result was seen.

### What did not work (ablations)
All numbers are keys30 tasks solved out of 30 unless marked "pp" (public LB points).

| Idea | Result | Decision |
|---|---|---|
| v3: budget forcing of the LLM's reasoning (`src3`) | dev-run replay 7.111 vs v2 8.778 | not submitted |
| v4: raise the token cap to 96k (E4 → c96) | offline adjusted 7.837 (+0.08 vs v2, a tie); public **29.03 (−3.19 pp)** | rejected for final selection |
| E6: v4 settings measured on the pre-registered 6-cell grid | raw 9.639 (SD 1.29) vs E3 (v2 settings) 8.278; adjusted 7.837 vs 7.753 (tie) | basis of the v4 decision above |
| C2: "medium" reasoning effort + execution-feedback repair (`src6`) | adjusted 6.778, Δ −0.975 vs v2; medium effort did not shorten reasoning, repairs were consumed by thinking | not submitted |
| C1 + ARC-1-style DSL search (`runs/arc-agi-2/v13/arc1/`) | zero-GPU replay adjusted 7.753 = v2 (no contribution) | dropped |
| v5: v2 with fresh seeds | public 33.89 (+1.67 pp, inside noise) | not selected (same family as slot 1) |
| v7 coverage scheduling, v9 confidence-ordered pass B, v11 half-cost width | public 30.14 / 29.72 / 30.56 & 28.89 | no clear gain over v8 |
| Public-notebook forks (perfpatch, Boyko V22) | public 30.14 / 30.56 | not selected |
| E1 (1× L4 GCP replay, 3 seeds): one full-width pass (F) vs half-width passes (HA, HA+HB pooled) | F 9.0 / 5.83 / 8.0; HA 6.667 / 7.0 / 6.667; pooled 7.5 / 7.5 / 7.167; pre-registered verdict "inconclusive" | no switch |
| Early pipelines (Qwen2.5-Coder baseline, Qwen3-2B grid model) | public 0.00 / 4.58 | abandoned for the 4B NVARC pipeline |

Details: `artifacts/verification/arc-agi-2/v13/prereg-*.md`, `e6-verdict-v1.txt`, `e4-verdict-v1.json`, `replay-*.json`, `gcp-e1-result-v1.json`, `new-approach-proposal-v1.md`.

### Repository layout
```
runs/arc-agi-2/
  v13/                 final selection 1 (program synthesis)
    src/               exact sources of the submitted v2 notebook (SHA-256 in preparation.json)
    src5/ src7/        v5 (seed/concurrency flags) / current template (adds model_profiles.py)
    src3/ src6/        v3 (budget forcing) / C2 (effort + repair) ablations
    upload-arc-2026-v13-progsynth-v2/   the submitted notebook + kernel-metadata (canonical)
    upload-arc-2026-v13-progsynth-v5-rs16/
    upload-arc-2026-v13-e3-seeds-v1/ -e4-cap96k-v1/ -e6-v4seeds-v1/   experiment notebooks (E3/E4/E6)
    build.py build_e3..e6.py audit_v13.py   notebook builders and post-run audit
    replay_*.py slot2_*.py offline_v13.py cap_yield.py e4_verdict.py c2_measure.py experiments/   offline analyses
    arc1/              ARC-1-style DSL experiment
    fetch/             source of the Python 3.12 fetch kernel
    tests*/            CPU tests (fake vLLM server)
  v6/ v7/ v8/ v9/ v10/ v11/   NVARC lineage (v8 = final selection 2); v6/author-v10 = upstream notebook cells
  p0-qwen/ p0b-colab/  Qwen3.8 probes (throughput, program-synthesis pilot)
  gcp-e1/ colab-cal-v1/ colab-pair-v1/   offline replay harness (1× L4 VM) and Colab calibration helpers
  fork-perfpatch/ fork-boyko/            public-notebook forks
  v6/output-*/         a few small outputs of the submitted commit runs
artifacts/verification/arc-agi-2/        pre-registrations, audits, hidden-shape results, replay results
tests/                 hidden-shape and pass-B tests for v6–v11 and the forks
```

### How to reproduce on Kaggle
- **Hardware / runtime:** Kaggle notebook, accelerator 4× NVIDIA L4 (`machine_shape: NvidiaL4`), internet **off**, 12 h limit (rerun budget `12*3600-600` s). Docker image pinned by digest in each `kernel-metadata.json` (`gcr.io/kaggle-private-byod/python@sha256:320043e1…`).
- **Attach (v13 v2):** competition `arc-prize-2026-arc-agi-2`; models `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1` and `mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1`; dataset `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`; kernels `sorokin/pip-install-unsloth-flash-patch` and our Python fetch kernel https://www.kaggle.com/code/f0909172434/arc-2026-py312-fetch-v1 (downloads python-build-standalone 3.12.14 and checks its SHA-256; source in `runs/arc-agi-2/v13/fetch/`).
- **Notebook:** `runs/arc-agi-2/v13/upload-arc-2026-v13-progsynth-v2/arc-2026-v13-progsynth-v2.ipynb` is the canonical submitted artifact. To run your own copy, change `id` in `kernel-metadata.json` to your username (the file is kept exactly as submitted, including `"is_private": true`; some tests and audits check these values) and run `kaggle kernels push -p runs/arc-agi-2/v13/upload-arc-2026-v13-progsynth-v2`, then submit that kernel version to the competition. For v8 use `runs/arc-agi-2/v8/upload-arc-2026-v8-4b-v2/`.
- **Modes:** the commit run (no `KAGGLE_IS_COMPETITION_RERUN`) is a development measurement: 4-task NVARC smoke test plus program-synthesis jobs on keys30, with the LLM phase ending 7,200 s after start (about 13.8k charged GPU-seconds including boot). The competition rerun runs the full pipeline on the hidden test file.
- **Rebuilding the notebook:** `python runs/arc-agi-2/v13/build.py <slug>` (needs `nbformat`). **Note:** the current `build.py` uses the post-2026-10-24 template, which also bundles `model_profiles.py` (from `src7/`). The submitted v2 notebook (SHA-256 `f93be126…`) predates this, so a rebuild is functionally equivalent but **not byte-identical**; use the upload folder when you need the exact artifact. v8: `python runs/arc-agi-2/v6/build.py arc-2026-v8-4b-v2 --src runs/arc-agi-2/v8/src`.
- **Not included:** our Kaggle launch/quota tooling (`launch_v13.py`, `v6/cloud.py`) is private infrastructure; `kaggle kernels push` replaces it. Two tests that exercised it are skipped (see below).

### Local checks (CPU)
```
pip install -r requirements.txt
# optional: competition files in data/arc-agi-2/official-v1/ (arc-agi_test_challenges.json, arc-agi_evaluation_challenges.json, ...)
python -m pytest runs/arc-agi-2/v13/tests
python -m pytest runs/arc-agi-2/v13/tests5
python -m pytest tests/test_arc_agi_2_v6_hidden_shape.py   # run test files one by one: the v8/v9/v11 tests import different src/ trees under the same module names
```
Status in our clean copy (Python 3.14, CPU): `v13/tests` 12 passed + 1 skipped (needs the unpublished launch layer); `v13/tests5` 10, `v13/tests3` 14, `v13/tests6` 13, `v13/tests_e4` 5 passed; `python runs/arc-agi-2/v13/tests_e6/test_e6.py` passes with its launch-gate checks skipped (it is a script; under pytest it misreads `sys.argv`, as in the original); every file in `tests/` passes when run on its own, except one pre-existing failure in `tests/test_arc_agi_2_fork_audit.py::test_all_log_files_concatenated` (also fails in the original private repo). `tests_e5/` was removed: it imported a module (`launch_v13_staged`) that never existed in the repository. Replays in `offline_v13.py`, `slot2_*.py`, `replay_*.py`, `arc1/replay_arc1.py` and `experiments/` need the E1 replay outputs (`runs/arc-agi-2/gcp-e1/out/e1/`, pickled decoder outputs), which are **not** included.

### Data
No ARC task files, Kaggle competition files or model weights are included. Get the tasks from the Kaggle competition data page or https://github.com/arcprize/ARC-AGI-2 and place them under `data/arc-agi-2/official-v1/`. `keys30.json` lists only 30 public-evaluation task IDs.

<a id="glossary"></a>
### Glossary: the multi-agent workflow
This project was run by AI coding agents (Claude Code sessions) under human supervision: the human owner set goals, approved spending and public actions, and made final calls on risky steps; agents wrote the code, ran experiments and reviewed one another. The records keep their vocabulary:

| Term | Meaning |
|---|---|
| owner session | the Claude Code session responsible for this competition ("ARC 對話" in the notes) |
| orchestrator / 協調者 | a separate coordinating session that allocated GPU quota across competitions, relayed public-notebook scans and kept backups |
| Argon | Gemini 4 Argon, used as an independent reviewer / drafting model |
| Opus | Claude Opus via the Anthropic API, used as an independent reviewer |
| Fable | Claude Fable, used as a reviewer for final-selection decisions |
| advisor | a stronger reviewer model that reads the whole session transcript before important decisions |
| DeepSeek `ds_agent.py` | DeepSeek model used to draft mechanical code edits that the owner then reviewed and applied |
| CSR | a "Claude Science" research request; e.g. CSR-arc-agi-2-001 proposed the program-synthesis channel |
| T0 / T1 / T2 / T3 | review tiers: T0 none; T1 one independent review; T2 (≥15,000 GPU-seconds or shared rules) two parallel blind reviews (Argon + Opus); T3 (final selection) Argon + Opus + advisor verdict + all submission gates |
| P0 / P1 / P2 | severity of review findings (P0 = must fix before launch) |
| R1, R4 | internal rules: R1 = exploratory submissions only need the automatic gates; R4 = final selection uses pre-registered offline estimates, a public-only candidate must beat the best offline-supported one by more than 2× the noise band |
| pre-registration (預登記, `prereg-*.md`) | a document written before a run that fixes the metric, the decision rule and the stopping rule, so the result cannot be reinterpreted after the fact |
| hidden-shape test | test that the notebook works on a test file with different task IDs, counts and order than the public one |
| HANDOFF | the private running log each session kept (not published) |
| settle | reconciling the GPU quota actually charged by Kaggle after a run |
| keys30, HA, HB, F, E1–E6, C1–C3 | 30 public-eval dev tasks; half-cost pass A / pass B; full-width pass; numbered experiments (see `artifacts/verification/arc-agi-2/v13/`) |

### License and attribution
Apache License 2.0 — see `LICENSE` and `NOTICE`. The NVARC pipeline files are derived from sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue (Apache-2.0); all upstream code, models, datasets and ideas are listed in `THIRD_PARTY.md`.

---

<a id="繁體中文"></a>
## 繁體中文

### 摘要
本 repo 是我們隊伍（Kaggle `f0909172434`）參加 **ARC Prize 2026 – ARC-AGI-2**（`arc-prize-2026-arc-agi-2`，2026-11-02 23:59 UTC 截止）的程式與實驗紀錄。最佳的最終選定提交先跑一輪公開的 NVARC 式 Qwen3-4B 測試時訓練流程（來自 sorokin 以 Apache-2.0 釋出的 notebook，我們加固過），再釋放 4 張 L4 GPU，讓 vLLM 服務的 **Qwen3.8-27B-FP8** 撰寫 Python `transform(grid)` 程式；只有能重現全部增強訓練對的程式，其答案才會被採用。公開榜分數 **32.22**。第二個最終選定是 NVARC 流程的兩輪版本（公開 **30.97**）。之後的多數想法都先預登記、離線量測，再依結果淘汰，數字都記錄在這裡。本專案由 AI 程式代理在人類監督下執行（見[術語表](#術語表)）。

### 成績
公開榜分數是隱藏測試輸出中解對的比例（pass@2），只計隱藏集的公開那一半。**本 repo 沒有私榜分數的紀錄。**

| Kaggle 提交 | 版本 | 公開榜 | 備註 |
|---|---|---:|---|
| **57165541** | **v13 v2**：NVARC pass A＋Qwen3.8-27B-FP8 程式合成 | **32.22** | **最終選定 1** |
| **56864814** | **v8**：NVARC pass A＋換種子的 pass B，合併候選 | **30.97** | **最終選定 2**（避險：不同種子與家族） |
| 57828532 | v13 v5：v2 程式換新種子、固定並行 16 | 33.89 | 與選定 1 同族，+1.67 pp 在 2.4 pp 雜訊帶內 → 不選 |
| 57214357 | v13 v2 重送（同一 kernel 版本） | 32.22 | 分數相同：評分重跑近乎確定性 |
| 57871700 | evgendvorkin V22（「Boyko」）的 fork | 30.56 | 低於當時銅牌線 → 不換入 |
| 57001190 / 57064408 | v11（半成本 TTT 寬度）r1 / r2 | 30.56 / 28.89 | |
| 56894434 / 56935323 | v8 重送 r2 / r3 | 29.72 / 29.72 | v8 三次平均 30.14 |
| 56917114 | v9（依把握度排序的 pass B） | 29.72 | |
| 56809468 / 56837247 | v6（加固的作者流程）／ v7（覆蓋率排程） | 30.14 / 30.14 | |
| 57122419 | mikelou1「perfpatch」原樣 fork | 30.14 | |
| 57301011 | v13 v4-c96（token 上限 96k） | 29.03 | 比 v2 低 3.19 pp |
| 56761143 | 早期流程：Qwen3-2B 格子模型 | 4.58 | 已淘汰（程式未收錄） |
| 56751754 | 早期基線：Qwen2.5-Coder-7B | 0.00 | 已淘汰（程式未收錄） |

- 銅牌線：2026-11-02 約 18:40 UTC 從即時公開榜讀取（3,627 隊，第 362 名）為 **32.22**。因此我們的 32.22 在公開榜上正好落在銅牌線；截止時的最終公開名次沒有紀錄。
- 雜訊：同家族重跑的公開分標準差約 0.90 pp；公開分差 ≤ 2.4 pp 視為雜訊。最終選定依預登記的離線估計決定，而不是看公開榜。
- 離線估計（30 題公開評估題「keys30」，以解對題數／30 計，預登記格點為 2 次重複 × 3 個種子）：v13 v2 原始 8.278、依佇列順序調整後 **7.753**；v8 為 **7.61**（3 個種子）。單獨的 NVARC 半成本 pass A（「HA」）為 6.778。

來源：`artifacts/verification/arc-agi-2/v13/`（預登記、重播、第 2 名額分析）、`artifacts/verification/arc-agi-2/v6/audit-*.json`、`artifacts/verification/arc-agi-2/v13-offline-v1.json`、`artifacts/verification/arc-agi-2/gcp-e1-result-v1.json`。

### 方法
**第 1 階段：NVARC pass A（Qwen3-4B）。** 程式在 `runs/arc-agi-2/v11/src/`（原封不動複製到 `runs/arc-agi-2/v13/src/`）。每一題：以增強後的訓練範例對 `sorokin/qwen3_4b_grids15_sft139` 做 rank-256 rsLoRA 測試時訓練（半成本寬度：8 種訓練增強 × 8 種幾何變換 ＝ 64 條序列），16 視角 turbo-DFS 解碼，增強重評分，再以 `kgmon` 選出前兩名候選。我們對作者 notebook 的修改（v6 起）：自動偵測路徑；commit 模式只跑 4 題冒煙測試、評分重跑模式跑全部題目；開始前先寫出完整的備援 `submission.json`；單題例外隔離；每張 GPU 一個獨立 worker 程序；原子寫檔；到期強制收尾。v13 以環境變數關閉第二輪 NVARC（v8 的「pass B」）。

**第 2 階段：Qwen3.8-27B-FP8 程式合成（`runs/arc-agi-2/v13/src/v13_llm.py`、`progsynth.py`）。** pass A 組成合法的 submission 後釋放 GPU，在獨立的 Python 3.12 虛擬環境以 vLLM 0.19.0 服務 Qwen3.8-27B-FP8（張量平行 4、MTP 推測解碼 2 個 token、fp8 KV cache）。模型負責撰寫 `transform(grid)`；每個候選程式在沙箱子程序中執行，只有重現**全部增強訓練對**才算通過驗證。

**合併規則（預登記，`runs/arc-agi-2/v13/src/v13_merge.py`）。** 每個測試輸出：已驗證程式的答案（多數決，平手取最先驗證者）放 `attempt_1`，NVARC 第一名放 `attempt_2`。如果程式答案已是 NVARC 兩個答案之一，就不更動；如果 NVARC 沒有真正的格子，`attempt_2` 用下一個不同的程式答案。每通過一個程式就原子改寫 submission；第 2 階段任何失敗都保留 NVARC 的 submission。

**v8（最終選定 2）。** 同一 NVARC 流程（作者原始寬度），加上 pass B：利用 pass A 剩下的時間，以獨立種子重跑題目，再把 pass A 與 pass B 的候選合併後選答（`runs/arc-agi-2/v8/src/arc_v6_config.py`）。

**工程做法。** 每次提交都要通過隱藏資料形狀測試（題目 ID、題數、順序都和公開檔不同，並注入 worker 當機或卡住；例如 `tests/test_arc_agi_2_v6_hidden_shape.py`）、commit 執行後的稽核（`runs/arc-agi-2/v13/audit_v13.py`、`runs/arc-agi-2/v6/audit.py`）；GPU 用量較大的實驗另外先寫預登記，在看到結果前固定判定規則。

### 沒有成功的嘗試（消融）
除標示「pp」（公開榜分數點）外，數字皆為 keys30 解對題數／30。

| 想法 | 結果 | 決定 |
|---|---|---|
| v3：強制截斷 LLM 推理預算（`src3`） | 開發執行重播 7.111，v2 為 8.778 | 不提交 |
| v4：token 上限提高到 96k（E4 → c96） | 離線調整後 7.837（比 v2 +0.08，持平）；公開 **29.03（−3.19 pp）** | 不列入最終選定 |
| E6：在預登記的 6 格上量測 v4 設定 | 原始 9.639（SD 1.29），E3（v2 設定）為 8.278；調整後 7.837 vs 7.753（持平） | 上面 v4 判定的依據 |
| C2：「medium」推理強度＋依執行結果修補（`src6`） | 調整後 6.778，比 v2 Δ −0.975；medium 沒有縮短推理，修補額度被思考吃光 | 不提交 |
| C1＋ARC-1 式 DSL 搜尋（`runs/arc-agi-2/v13/arc1/`） | 零 GPU 重播調整後 7.753，與 v2 相同（零貢獻） | 放棄 |
| v5：v2 換新種子 | 公開 33.89（+1.67 pp，在雜訊內） | 不選（與名額 1 同族） |
| v7 覆蓋率排程、v9 依把握度排序 pass B、v11 半成本寬度 | 公開 30.14／29.72／30.56 與 28.89 | 相對 v8 沒有明確增益 |
| 公開 notebook fork（perfpatch、Boyko V22） | 公開 30.14／30.56 | 不選 |
| E1（GCP 1× L4 重播，3 個種子）：全寬單輪（F）vs 半寬（HA、HA＋HB 合併） | F 9.0／5.83／8.0；HA 6.667／7.0／6.667；合併 7.5／7.5／7.167；預登記結論「無法判定」 | 不切換 |
| 早期流程（Qwen2.5-Coder 基線、Qwen3-2B 格子模型） | 公開 0.00／4.58 | 改用 4B NVARC 流程 |

細節：`artifacts/verification/arc-agi-2/v13/prereg-*.md`、`e6-verdict-v1.txt`、`e4-verdict-v1.json`、`replay-*.json`、`gcp-e1-result-v1.json`、`new-approach-proposal-v1.md`。

### 目錄結構
```
runs/arc-agi-2/
  v13/                 最終選定 1（程式合成）
    src/               送出的 v2 notebook 的確切原始碼（SHA-256 記在 preparation.json）
    src5/ src7/        v5（種子／並行參數）／ 目前模板（多了 model_profiles.py）
    src3/ src6/        v3（預算截斷）／ C2（推理強度＋修補）消融
    upload-arc-2026-v13-progsynth-v2/   送出的 notebook＋kernel-metadata（正式版本）
    upload-arc-2026-v13-progsynth-v5-rs16/
    upload-arc-2026-v13-e3-seeds-v1/ -e4-cap96k-v1/ -e6-v4seeds-v1/   實驗 notebook（E3／E4／E6）
    build.py build_e3..e6.py audit_v13.py   notebook 建置器與執行後稽核
    replay_*.py slot2_*.py offline_v13.py cap_yield.py e4_verdict.py c2_measure.py experiments/   離線分析
    arc1/              ARC-1 式 DSL 實驗
    fetch/             Python 3.12 下載 kernel 的原始碼
    tests*/            CPU 測試（假的 vLLM 伺服器）
  v6/ v7/ v8/ v9/ v10/ v11/   NVARC 系列（v8＝最終選定 2）；v6/author-v10＝上游 notebook 的 cell
  p0-qwen/ p0b-colab/  Qwen3.8 探測（吞吐量、程式合成先導）
  gcp-e1/ colab-cal-v1/ colab-pair-v1/   離線重播工具（1× L4 VM）與 Colab 校準輔助
  fork-perfpatch/ fork-boyko/            公開 notebook 的 fork
  v6/output-*/         送出版本 commit 執行的少量小型輸出
artifacts/verification/arc-agi-2/        預登記、稽核、隱藏形狀測試結果、重播結果
tests/                 v6–v11 與 fork 的隱藏形狀與 pass B 測試
```

### 如何在 Kaggle 重現
- **硬體與執行時間：** Kaggle notebook，加速器 4× NVIDIA L4（`machine_shape: NvidiaL4`），**關閉**網路，12 小時上限（評分重跑預算 `12*3600-600` 秒）。Docker 映像以 digest 固定在各 `kernel-metadata.json`（`gcr.io/kaggle-private-byod/python@sha256:320043e1…`）。
- **需掛載（v13 v2）：** 競賽 `arc-prize-2026-arc-agi-2`；模型 `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`、`mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1`；資料集 `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`；kernel `sorokin/pip-install-unsloth-flash-patch` 與我們的 Python 下載 kernel https://www.kaggle.com/code/f0909172434/arc-2026-py312-fetch-v1（下載 python-build-standalone 3.12.14 並檢查 SHA-256；原始碼在 `runs/arc-agi-2/v13/fetch/`）。
- **Notebook：** `runs/arc-agi-2/v13/upload-arc-2026-v13-progsynth-v2/arc-2026-v13-progsynth-v2.ipynb` 是送出時的正式版本。要跑自己的副本，把 `kernel-metadata.json` 的 `id` 改成你的帳號（檔案保持送出時原樣，包括 `"is_private": true`；部分測試與稽核會檢查這些值），執行 `kaggle kernels push -p runs/arc-agi-2/v13/upload-arc-2026-v13-progsynth-v2`，再把該 kernel 版本提交到競賽。v8 請用 `runs/arc-agi-2/v8/upload-arc-2026-v8-4b-v2/`。
- **模式：** commit 執行（沒有 `KAGGLE_IS_COMPETITION_RERUN`）是開發量測：4 題 NVARC 冒煙測試，加上 keys30 的程式合成工作，LLM 階段在開始後 7,200 秒結束（含開機約扣 13.8k GPU 秒）。競賽評分重跑則對隱藏測試檔跑完整流程。
- **重建 notebook：** `python runs/arc-agi-2/v13/build.py <slug>`（需要 `nbformat`）。**注意：** 目前的 `build.py` 使用 2026-10-24 之後的模板，會多打包 `model_profiles.py`（來自 `src7/`）。送出的 v2 notebook（SHA-256 `f93be126…`）早於這次修改，所以重建結果功能相同，但**不會逐位元組相同**；需要確切版本時請用 upload 資料夾。v8：`python runs/arc-agi-2/v6/build.py arc-2026-v8-4b-v2 --src runs/arc-agi-2/v8/src`。
- **未收錄：** 我們的 Kaggle 發射與額度工具（`launch_v13.py`、`v6/cloud.py`）屬於私人基礎設施，可用 `kaggle kernels push` 取代。用到它的兩個測試已改為略過（見下）。

### 本機檢查（CPU）
```
pip install -r requirements.txt
# 選用：把競賽檔放在 data/arc-agi-2/official-v1/（arc-agi_test_challenges.json、arc-agi_evaluation_challenges.json 等）
python -m pytest runs/arc-agi-2/v13/tests
python -m pytest runs/arc-agi-2/v13/tests5
python -m pytest tests/test_arc_agi_2_v6_hidden_shape.py   # 請逐檔執行：v8／v9／v11 的測試以相同模組名稱載入不同的 src/
```
在我們的乾淨副本（Python 3.14、CPU）上：`v13/tests` 12 項通過、1 項略過（需要未公開的發射層）；`v13/tests5` 10 項、`v13/tests3` 14 項、`v13/tests6` 13 項、`v13/tests_e4` 5 項通過；`python runs/arc-agi-2/v13/tests_e6/test_e6.py` 通過，但略過發射閘門檢查（它是腳本；用 pytest 跑會誤讀 `sys.argv`，原 repo 也一樣）；`tests/` 的每個檔案單獨執行都通過，只有 `tests/test_arc_agi_2_fork_audit.py::test_all_log_files_concatenated` 一項原本就失敗（在原本的私人 repo 也失敗）。`tests_e5/` 已移除：它 import 的模組（`launch_v13_staged`）從未存在於 repo 中。`offline_v13.py`、`slot2_*.py`、`replay_*.py`、`arc1/replay_arc1.py`、`experiments/` 的重播需要 E1 重播輸出（`runs/arc-agi-2/gcp-e1/out/e1/`，pickle 格式的解碼結果），本 repo **未收錄**。

### 資料
本 repo 不含任何 ARC 題目檔、Kaggle 競賽檔或模型權重。請從 Kaggle 競賽資料頁或 https://github.com/arcprize/ARC-AGI-2 取得題目，放在 `data/arc-agi-2/official-v1/`。`keys30.json` 只列出 30 個公開評估題的 ID。

<a id="術語表"></a>
### 術語表：多代理工作流程
本專案由 AI 程式代理（Claude Code 對話）在人類監督下執行：人類擁有者設定目標、核准花費與公開動作，並對有風險的步驟做最後決定；代理負責寫程式、跑實驗、互相審查。紀錄中保留了這些用語：

| 用語 | 意思 |
|---|---|
| owner session（ARC 對話） | 負責本競賽的 Claude Code 對話 |
| orchestrator／協調者 | 另一個協調用的對話，負責在各競賽間分配 GPU 額度、轉達公開 notebook 掃描結果、管理備份 |
| Argon | Gemini 4 Argon，作為獨立審查者與草稿模型 |
| Opus | 經 Anthropic API 呼叫的 Claude Opus，作為獨立審查者 |
| Fable | Claude Fable，用於最終選定決策的審查 |
| advisor | 在重要決定前讀完整個對話紀錄的較強審查模型 |
| DeepSeek `ds_agent.py` | 用 DeepSeek 模型起草機械性的程式修改，再由擁有者審查後套用 |
| CSR | 「Claude Science」研究請求；例如 CSR-arc-agi-2-001 提出了程式合成這條路線 |
| T0／T1／T2／T3 | 審查分級：T0 不審；T1 一份獨立審查；T2（≥15,000 GPU 秒或共用規則）兩份平行盲審（Argon＋Opus）；T3（最終選定）Argon＋Opus＋advisor 裁決＋全部提交閘門 |
| P0／P1／P2 | 審查發現的嚴重度（P0＝發射前必修） |
| R1、R4 | 內部規則：R1＝探索性提交只需通過自動閘門；R4＝最終選定依預登記的離線估計，只有公開分的候選必須比有離線支持的最佳版本高出 2 倍雜訊帶以上 |
| 預登記（pre-registration，`prereg-*.md`） | 在執行前寫好的文件，固定指標、判定規則與停止規則，避免看到結果後再重新解讀 |
| 隱藏形狀測試 | 確認 notebook 在題目 ID、題數、順序都和公開檔不同的測試檔上也能正常執行 |
| HANDOFF | 各對話維護的私人工作日誌（未公開） |
| settle | 執行後對帳 Kaggle 實際扣掉的 GPU 額度 |
| keys30、HA、HB、F、E1–E6、C1–C3 | 30 題公開評估開發題；半成本 pass A／pass B；全寬單輪；編號的實驗（見 `artifacts/verification/arc-agi-2/v13/`） |

### 授權與署名
Apache License 2.0，見 `LICENSE` 與 `NOTICE`。NVARC 流程相關檔案衍生自 sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue（Apache-2.0）；所有上游程式、模型、資料集與想法都列在 `THIRD_PARTY.md`。
