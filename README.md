# ARC Prize 2026 (ARC-AGI-2): how test-time training and verified program synthesis work together

[English](#english) | [繁體中文](#繁體中文)

---

<a id="english"></a>
## English

This repository is a worked, end-to-end explanation of a complete Kaggle solution to **ARC Prize 2026 – ARC-AGI-2**, with all the code that ran. It is written for people who want to learn *why* the parts work, not only what we submitted. Every section points to the file that implements it.

**In one paragraph.** A 4-billion-parameter language model that has been pre-trained on ARC-style grids is **fine-tuned separately on each puzzle at test time** (a fresh LoRA adapter per task, trained only on that task's 2–5 examples under symmetry and colour augmentations). It then enumerates every output grid it considers likely and keeps the ones that stay likely when the puzzle is rotated, reflected or recoloured. After this pass, a much larger model (Qwen3.8-27B) is asked to **write a Python program** for each puzzle. A program is trusted only if it reproduces every training example of a disguised (rotated + recoloured) copy of the puzzle. The two channels fail in different ways — the first proposes many plausible grids, the second rarely answers but is usually right when it does — and a simple pre-registered rule combines them under the competition's two-attempt scoring.

Contents: [1. The task](#1-the-task-and-why-it-is-hard) · [2. Pipeline](#2-the-pipeline-at-a-glance) · [3. Test-time training](#3-test-time-training-a-new-adapter-for-every-puzzle) · [4. Decoding](#4-decoding-enumerate-every-likely-grid) · [5. Augmentation re-scoring](#5-choosing-candidates-consistency-across-symmetries) · [6. Program synthesis](#6-program-synthesis-with-verification) · [7. Merging](#7-merging-two-channels-under-pass2) · [8. Engineering for a hidden rerun](#8-engineering-for-a-hidden-12-hour-rerun) · [9. Measuring progress](#9-measuring-progress-when-the-leaderboard-is-noisy) · [10. What did not work, and why](#10-what-did-not-work-and-why) · [11. Results](#11-results) · [12. Reproduce](#12-reproduce) · [Glossary](#glossary)

### 1. The task and why it is hard
An ARC task shows a few **input → output grid pairs** (grids up to 30×30, cells coloured 0–9) that all follow one hidden rule, plus one or more test inputs. The solver must produce the exact output grid: every cell and the grid size must match. A toy example (not from the competition data):

```
example 1:  1 0 0      0 0 1        rule: mirror left-right
            2 0 0  ->  0 0 2
            0 0 0      0 0 0
test:       0 3 0  ->  ?           answer: 0 3 0 / 4 0 0 / 0 0 0  (mirrored)
            0 0 4
            0 0 0
```

What makes ARC-AGI-2 hard for machine learning:
- **Almost no data per rule.** Each task has 2–5 examples, and the hidden test tasks use rules that do not appear in the training set. A model cannot memorise; it has to infer the rule from the examples in front of it.
- **Exact-match scoring.** One wrong cell is a wrong answer. Two attempts per test output are allowed (**pass@2**); the score is the share of test outputs solved.
- **Hard compute limits.** Kaggle reruns the notebook on hidden tasks with **no internet, 4× NVIDIA L4 GPUs and 12 hours** in total. Everything — fine-tuning, search, a 27B model — must fit inside that budget.

### 2. The pipeline at a glance
```
hidden test tasks
   │
   ├─ Phase 1: NVARC pass A (Qwen3-4B, 4 GPUs, one worker per GPU)        runs/arc-agi-2/v13/src/arc_solver.py
   │     per task: test-time LoRA training → DFS enumeration → augmentation re-scoring → top-2 grids
   │     → a complete, valid submission.json is written first                    (assemble.py)
   │
   ├─ Phase 2: program synthesis (Qwen3.8-27B-FP8 on vLLM, tensor-parallel 4) v13_llm.py, progsynth.py
   │     per task, least-confident first: write transform(grid) → run in sandbox on a disguised copy
   │     → accepted only if every training pair is reproduced
   │
   └─ Merge (pre-registered): verified program answer → attempt_1, NVARC top-1 → attempt_2   v13_merge.py
```
The test-time-training pipeline is not ours. It is the public **NVARC**-style notebook by sorokin (Apache-2.0), built on a Qwen3-4B model that sorokin pre-trained on ARC-like grids (`sorokin/qwen3_4b_grids15_sft139`). Our contributions are the program-synthesis channel, the merge rule, the reliability work in §8 and the measurement work in §9; `THIRD_PARTY.md` lists every upstream source.

### 3. Test-time training: a new adapter for every puzzle
*Code: `solve_task()` and `worker()` in `runs/arc-agi-2/v13/src/arc_solver.py`; augmentations in `arc_loader.py`.*

**Idea.** A model pre-trained on many ARC-like puzzles has useful general priors (objects, symmetry, counting, colour roles), but the rule of a new puzzle is not in its weights. Test-time training (TTT) puts it there: before answering, the model takes a few gradient steps on *this puzzle's own examples*. Over-fitting is the goal — the model only has to be right about one rule.

**How the training data is made from 2–5 examples.** ARC rules are almost always unchanged by rotating/reflecting the whole puzzle, renaming colours, or reordering the examples. Each of these gives a new, equally valid training sequence (`ArcDataset.augment`):
- the 8 symmetries of a square (identity, 3 rotations, transpose and its 3 rotations): `transpose` × `rot90` applied to *every* grid of the task;
- a random permutation of the 10 colours (`permute_mod`), applied consistently to inputs and outputs — drawn independently for every symmetry copy, so 8 symmetries × 8 draws give 64 differently coloured sequences;
- a shuffle of the example order (`shuffle_ex`), also drawn per sequence.

One training sequence is the whole task written as a chat: `user: <input grid> / assistant: <output grid>` for each example, with grids written as digit rows (`QwenFormatter`). The loss is computed **only on the assistant (output-grid) tokens** (`QwenDataCollatorForCompletionOnlyLM` masks everything else), so the model learns "given these examples and this input, produce this output", not to reproduce inputs.

**Settings and why.** LoRA rank 256 with rank-stabilised scaling (rsLoRA) on every attention and MLP projection plus the embedding and output layers; learning rate 5e-5, cosine schedule, 1 epoch, bf16. The adapter is reset to the same initial weights before each task (`set_peft_model_state_dict(... default_weights)`), so tasks never leak into each other. The upstream notebook uses 16 colour permutations × 8 symmetries = 128 sequences per task; our submitted v13 uses a **half-cost width** of 8 × 8 = 64 (`N_TRAIN_AUG=8` in `arc_v6_config.py`) to free GPU time for phase 2. Sequences longer than 8,192 tokens drop examples until they fit (`cut_to_len`).

### 4. Decoding: enumerate every likely grid
*Code: `turbo_dfs()` / `inference_turbo_dfs()` in `arc_solver.py`.*

After TTT, the model sees the test input (again under several symmetry/colour "views") and must write the output grid. Instead of sampling or beam search, the decoder does a **depth-first search over the whole output space, pruned by probability**:
- The vocabulary is restricted to 12 tokens: the digits 0–9, newline and end-of-message. With so few branches per step an exhaustive search is affordable.
- A partial grid is extended only while its cumulative negative log-likelihood stays below `max_score = −ln 0.2`. The search therefore returns the complete grids whose probability is at least 20% — usually one to a few candidates — and nothing else. (It is exhaustive only if it finishes: each search is also cut off at 540 s and at the task deadline.)
- With `N_EVAL_AUG=1` each test input is decoded in **8 views** (the 8 symmetries, each with its own colour permutation); the upstream full width uses 16.
- Views of one test input are decoded together in batches of up to 4 equal-length prompts (`decode_batches` in `arc_v6_config.py`); each recursion step is one batched forward pass that reuses the KV cache.

Why this instead of sampling: exact-match scoring rewards the single most probable grid, and a probability threshold gives a complete, deterministic candidate list with no temperature to tune. Each candidate is mapped back to the original orientation and colours (`invert_mod`).

### 5. Choosing candidates: consistency across symmetries
*Code: `solve_task()` (re-scoring loop) in `arc_solver.py`; `score_kgmon` in `arc_decoder.py`.*

A wrong answer can be confident in one view by accident. A right answer should stay likely in **every** view, because the rule is symmetric. So every distinct candidate is re-scored: the task (with the candidate as the test output) is augmented into 8 new views (all 8 symmetries, each with a fresh colour permutation and example order), and the model's negative log-likelihood of the whole candidate grid — the **sum** over its tokens — is computed in each view (`calc_scores`). Candidates are then ranked by the `kgmon` score:

```
kgmon(candidate) = (number of decoded views that produced this grid) − mean NLL of the grid over the 8 re-scoring views
```

The first term rewards agreement between independent decodings; the second rewards a grid that remains probable after the puzzle is disguised. The top two become `attempt_1` and `attempt_2`. Re-scoring results are cached per grid, so repeated candidates cost nothing extra.

### 6. Program synthesis with verification
*Code: `runs/arc-agi-2/v13/src/v13_llm.py` (orchestration), `progsynth.py` (prompt, sandbox, check), `arc_llm.py` (augmentation).*

**Idea.** A grid model can be close but wrong; a *program* either reproduces the examples or it does not. If a program reproduces every training pair, its test output is much more likely to be right than an unverified guess. Verification turns a low-recall generator into a high-precision one.

**Steps for one job.**
1. **Disguise the task** (`arc_llm.augmentation`, `augment_task`): one deterministic non-identity symmetry and a permutation of colours 1–9 (0 stays the background). This makes it unlikely that the model recognises a puzzle it has seen and recites a memorised answer, and it gives each repetition a different view.
2. **Prompt** (`progsynth.build_prompt`): the examples and test inputs as JSON-like rows, and a request for a complete `transform(grid)` in one Python block. Qwen3.8-27B-FP8 runs with thinking enabled (temperature 1.0, top-p 0.95, top-k 20), served by vLLM 0.19.0 with tensor parallelism over the 4 GPUs, MTP speculative decoding and an fp8 KV cache.
3. **Collect candidates**: the last code block of the final answer, then the last two code drafts inside the reasoning. Drafts cost no extra tokens and matter when a long reasoning trace is cut off by the token cap.
4. **Verify in a sandbox** (`progsynth.run_program`, `check`): a separate Python process per program with memory (2 GB), CPU-time (20 s) and file-size limits, an empty temporary working directory, stdout redirected away from the result channel, and the whole process group killed on timeout. *Verified* means every training output is reproduced exactly and every test output is a valid grid.
5. **Reject cheating programs** (`suspicious()` in `v13_llm.py`): a training output (≥ 9 cells, ≥ 2 colours) hard-coded in the source, a test answer (≥ 9 cells) that copies a training output, or a test answer equal to the test input when no training pair is the identity.
6. **Undo the disguise** on the test answers (`invert_aug`) and rewrite the submission atomically.

**Scheduling.** In the hidden rerun, tasks whose NVARC top-1 kgmon score is below 5.0 ("unsure") are queued first, least confident first; the "sure" tasks follow in file order (a task NVARC is already sure about gains little from a program). Concurrency is bounded by KV-cache memory: `int(kv_tokens / (0.35 × cap + 6000)) + 2` streams (`v13_llm.py`), 13 at the 64k cap. A task without a verified program is re-queued at the back with a new disguise, up to 6 repetitions, so a task is never in flight twice and later repetitions go only to unsolved tasks.

**What it achieves, honestly.** In our 2-hour development run on 30 public-evaluation tasks, 26 jobs started; only 2 finished naturally, 11 hit the 64k-token cap and 13 were stopped at the time limit. 4 programs were verified and 3 of them were correct (2 of the 4 verified programs were drafts recovered from reasoning that hit the cap). Merged with NVARC this added **+2.0 tasks out of 30** in each of the three replay seeds (6.67 → 8.67 for seed 1; `artifacts/verification/arc-agi-2/v13/replay-v2-merge.json`). The numbers are small, but they show the shape of the channel: high precision, and throughput bound by reasoning length (the two jobs that finished naturally used 47k and 56k tokens). That bound motivated most of the experiments in §10.

**Four jobs from that run, end to end** (task IDs from the public evaluation set):
- `dbff022c` — reasoning finished (47k tokens), program verified and correct; NVARC had no correct answer → the program answer took attempt_1: **+1 task**.
- `332f06d7` — reasoning hit the 64k cap, but a draft inside the reasoning verified and was correct → **+1 task**. Without draft extraction this job would have produced nothing.
- `2b83f449` — also hit the cap; a draft passed every training pair but was **wrong on the test input** (a false positive). It displaced NVARC's attempt_2, which was also wrong, so nothing was lost here — but this is exactly the case where the merge rule can cost a point.
- `53fb4810` — verified; its answer was already one of NVARC's two attempts → the entry stayed unchanged.

### 7. Merging two channels under pass@2
*Code: `fill_entry()` / `ranked_program_answers()` in `runs/arc-agi-2/v13/src/v13_merge.py`.*

For each test output: the answer supported by the most verified programs (ties → the first verified) becomes `attempt_1`; NVARC's top-1 becomes `attempt_2`. If the program answer already equals one of NVARC's two attempts, nothing changes. If NVARC produced no real grid, `attempt_2` is the next distinct program answer.

Why this order: with two attempts, the only cost of inserting the program answer is losing NVARC's *second* choice. A verified program is far more often right than NVARC's runner-up, so displacing attempt_2 is usually cheap — but not free: a false-positive program (like `2b83f449` above) displaces attempt_2, and if that attempt_2 had been correct the task would be lost. In the replay, program answers changed 3 outputs per seed (seeds 1–2: three displaced attempt_2; seed 3: two displaced attempt_2 and one filled an empty NVARC entry); 2 per seed became correct and no correct NVARC answer was lost (`lost_correct_nvarc: false` in every row). The rule was fixed in writing before the measurement.

### 8. Engineering for a hidden 12-hour rerun
*Code: `runs/arc-agi-2/v13/src/assemble.py`, `arc_v6_config.py`, `arc_solver.py` (`worker`), `v13_llm.py`.*

A Kaggle code-competition notebook is rerun on data you never see. A crash at hour 11 scores zero. The defences, and the failure each one prevents:
- **Write a complete fallback first.** A valid `submission.json` (placeholder grids for every test output) exists before any model loads; NVARC results and later every verified program overwrite it **atomically** (`write → os.replace`), so a kill at any moment leaves a valid file.
- **Isolate failures.** One process per GPU; an exception in one task is logged and the worker moves on; a worker retires after 3 consecutive failures, where an out-of-memory error counts half (so 6 OOMs in a row); a sticky CUDA fault stops only that worker. Unreadable or half-written results are skipped, not fatal.
- **Budget the wall clock.** A global deadline (12 h − 10 min); a per-task decode cap that shrinks to a fair share of the remaining time (`adaptive_cap`) but never below 360 s, and no new task is started with less than 360 s left; test-time training itself is not time-limited, and every test input gets at least one decode batch unless the global deadline has passed.
- **Phase 2 fails safe.** Any failure before the first program job (missing inputs, GPUs still busy, install, server start) skips phase 2 and keeps NVARC's submission unchanged; after that, only verified programs touch the file. (Its *answers* can still cost a point through the merge rule, §7.) vLLM runs from its own Python 3.12 environment so the NVARC stack is never touched; the interpreter archive and the vLLM and torch wheels are checked against SHA-256 values pinned in the code, and the rest of the wheelhouse against its `SHA256SUMS` file.
- **Test the hidden shape.** Before every submission, CPU tests run the notebook logic on test files with different task IDs, counts and order, with injected crashes and hangs (`tests/test_arc_agi_2_v6_hidden_shape.py`, `runs/arc-agi-2/v13/tests*/`).

### 9. Measuring progress when the leaderboard is noisy
- **The public score is a noisy sample.** Re-submitting the same family with new seeds moved the public score with a standard deviation of about 0.9 points; we treated differences up to **2.4 points** as noise. The public leaderboard also covers only part of the hidden set. Picking the final submissions by the highest public score would mostly pick luck.
- **A fixed offline set.** We measured on 30 public-evaluation tasks ("keys30", `keys30.json`) with a pre-registered grid of repetitions and seeds, and reported tasks solved out of 30.
- **Cache the expensive stage, replay the cheap one.** NVARC outputs for keys30 were generated once per seed on a single-GPU VM (experiment E1) and saved; any new merge rule or program set can then be scored on them with zero GPU (`runs/arc-agi-2/v13/offline_v13.py`, `replay_*.py`).
- **Adjust for capacity.** In the hidden rerun the same LLM time is shared by several times more tasks than keys30, so not every task gets the attempts it got offline. The program channel's whole gain over NVARC alone was therefore scaled by a pre-registered capacity factor c: `adjusted = HA + c × (raw − HA)`, where HA = 6.778 is half-width NVARC alone, c = 0.65 at the 64k-token cap and 0.37 at 96k (`prereg-e6-v4-seeds-v1.md`, `c2_measure.py`). The numbers that appear in this README:

  | quantity | v13 v2 |
  |---|---:|
  | HA: NVARC half-width alone (E1, mean of 3 seeds) | 6.778 |
  | single development run merged with HA (`v2_dev`) | 8.778 |
  | raw: mean of the pre-registered repetitions (E3) | 8.278 |
  | **adjusted**: 6.778 + 0.65 × (8.278 − 6.778) | **7.753** |
- **Pre-register.** Before each larger run we wrote the metric, the decision rule and the stopping rule (`artifacts/verification/arc-agi-2/v13/prereg-*.md`), so a result could not be reinterpreted after it was seen.

### 10. What did not work, and why
| Idea | What we expected | What happened | Why (our reading) |
|---|---|---|---|
| Raise the token cap 64k → 96k (v4 / E4, E6) | More reasoning traces finish, more programs verify (a cap-yield model predicted about +2.3 public points; `cap-yield-v1.json`) | Raw offline gain (9.64 vs 8.28) vanished after the capacity adjustment (7.84 vs 7.75, a tie); one public reading **−3.19** | Longer jobs mean fewer jobs fit in the window and fewer parallel KV slots (13 → 10 streams), so on the hidden set's many tasks fewer tasks get an attempt; the capacity factor fell from 0.65 to 0.37. The pre-registration notes that the public drop cannot be split between this lower coverage and false-positive programs displacing correct NVARC answers. |
| Budget forcing of reasoning (v3) | Shorter thinking → more jobs | dev replay 7.11 vs 8.78 | Hard ARC rules need long exploration; cutting it lost more solutions than the extra jobs found. |
| "Medium" reasoning effort + one round of execution feedback (C2) | Shorter traces, plus repairs of near-miss programs | −0.98 tasks vs v2 (adjusted 6.778 vs 7.753); P(Δ>0) = 0 | More rows were logged (83 vs 26 in a similar window, repair turns included) but fewer programs verified (3 vs 4). Our working explanation — lower effort produced less careful programs, and repair turns spent their budget on more reasoning — was not isolated by a separate experiment. |
| ARC-1-style DSL search added to the program channel (C1) | Cheap extra solutions | zero contribution in replay (7.753 = v2) | Most likely the tasks it could solve were already solved by NVARC or the LLM. |
| One full-width NVARC pass vs two half-width passes pooled (E1, v8 / v11) | Diversity from a second pass helps | pre-registered verdict "inconclusive" | Differences were within seed noise at n = 30. |
| Grid-model baselines (Qwen2.5-Coder-7B, Qwen3-2B) | — | public 0.00 / 4.58 | Without ARC-specific pre-training and test-time training, the models did not solve hidden tasks. |

### 11. Results
Public leaderboard (pass@2, public part of the hidden set). Private scores are not recorded here.

| Submission | What | Public |
|---|---|---:|
| 57165541 | v13 v2: NVARC pass A + verified program synthesis — **final selection 1** | 32.22 |
| 56864814 | v8: NVARC pass A + re-seeded pass B, pooled — **final selection 2** (a different family as a hedge) | 30.97 |
| 57828532 | v13 v5 (same code, new seeds) | 33.89 (inside the noise band; same family as selection 1, not selected) |
| other NVARC variants and public forks | v6, v7, v9, v11, perfpatch, Boyko V22 | 28.89–30.56 |

On 2026-11-02 (about 18:40 UTC, 3,627 teams) the public bronze line was 32.22. The full submission history is in `artifacts/verification/arc-agi-2/`.

### 12. Reproduce
- **On Kaggle.** Notebook with 4× NVIDIA L4, internet off, 12 h. Attach the competition, `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`, `mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1`, the dataset `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`, the kernel `sorokin/pip-install-unsloth-flash-patch` and our Python 3.12 fetch kernel https://www.kaggle.com/code/f0909172434/arc-2026-py312-fetch-v1 (source in `runs/arc-agi-2/v13/fetch/`). The submitted notebook is `runs/arc-agi-2/v13/upload-arc-2026-v13-progsynth-v2/`: change `id` in `kernel-metadata.json` to your user name, `kaggle kernels push -p <folder>`, then submit that version. A commit run (no `KAGGLE_IS_COMPETITION_RERUN`) is a development measurement on 4 smoke tasks + keys30; the competition rerun runs everything on the hidden tasks. v8 is in `runs/arc-agi-2/v8/upload-arc-2026-v8-4b-v2/`.
- **Rebuild.** `python runs/arc-agi-2/v13/build.py <slug>` (needs `nbformat`). The current builder bundles `model_profiles.py` (added after submission), so a rebuild is functionally equivalent but not byte-identical to the submitted notebook (SHA-256 `f93be126…`).
- **CPU tests.** `pip install -r requirements.txt`, then `python -m pytest runs/arc-agi-2/v13/tests` and `tests5`; run the files in `tests/` one at a time (v8/v9/v11 tests import different `src/` trees under the same module names). In our clean copy all pass except one test that needs our unpublished launch layer (skipped) and one pre-existing failure in `tests/test_arc_agi_2_fork_audit.py::test_all_log_files_concatenated`. Replays need the E1 decoder outputs, which are not included.
- **Data.** No ARC task files, Kaggle files or model weights are included. Get the tasks from the competition page or https://github.com/arcprize/ARC-AGI-2 and place them in `data/arc-agi-2/official-v1/`.

### Repository layout
```
runs/arc-agi-2/v13/src/        submitted v13 v2 sources (NVARC files + v13_llm.py, progsynth.py, arc_llm.py, v13_merge.py)
runs/arc-agi-2/v13/            builders, audits, offline replays, ablation sources (src3, src5, src6, src7, arc1/)
runs/arc-agi-2/v6 … v11/       NVARC lineage (v8 = final selection 2; v6/author-v10 = upstream cells)
runs/arc-agi-2/gcp-e1/         single-GPU replay harness for offline measurement
artifacts/verification/arc-agi-2/   pre-registrations, audits, hidden-shape results, replay results
tests/                         hidden-shape and pass-B tests
```

<a id="glossary"></a>
### Glossary: how the project was run
The project was run by AI coding agents (Claude Code sessions) under human supervision: the human owner set goals, approved spending and public actions, and made the final call on risky steps; agents wrote code, ran experiments and reviewed one another. The records keep their vocabulary:

| Term | Meaning |
|---|---|
| owner session / orchestrator (協調者) | the session responsible for this competition / a coordinating session that allocated GPU quota and relayed public-notebook scans |
| Argon, Opus, Fable, advisor | independent reviewer models (Gemini 4 Argon; Claude Opus via API; Claude Fable for final selection; a stronger model that reads the whole transcript before key decisions) |
| T0–T3, P0–P2 | review tiers (none / one review / two blind reviews for large GPU spend or shared rules / full gate for final selection) and finding severity (P0 = must fix) |
| R1, R4 | internal rules: exploratory submissions need only automatic gates; final selection follows pre-registered offline estimates, and a public-only candidate must beat the best offline-supported one by more than 2× the noise band |
| pre-registration (`prereg-*.md`) | metric, decision rule and stopping rule written before a run |
| hidden-shape test | a test that the notebook works with different task IDs, counts and order |
| keys30, HA, HB, F, E1–E6, C1–C3 | the 30 dev tasks; half-cost pass A / pass B; full-width pass; numbered experiments |

### License and attribution
Apache License 2.0 — see `LICENSE` and `NOTICE`. The NVARC pipeline files derive from sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue (Apache-2.0); all upstream code, models, datasets and ideas are listed in `THIRD_PARTY.md`.

---

<a id="繁體中文"></a>
## 繁體中文

本 repo 完整解說一套 **ARC Prize 2026 – ARC-AGI-2** 的 Kaggle 解法，並附上實際執行的全部程式。寫給想了解各部分**為什麼有效**的讀者，而不只是列出我們交了什麼。每一節都會指出實作它的檔案。

**一段話說完。** 一個已在 ARC 類網格上預訓練過的 40 億參數語言模型，會在**測試時針對每一題各自微調**：每題一個全新的 LoRA adapter，只用該題的 2–5 個示範例，加上對稱與換色擴增來訓練。接著模型列舉它認為可能的所有輸出網格，只留下在題目被旋轉、翻轉或換色後仍然可能的那些。這一輪結束後，再請一個大得多的模型（Qwen3.8-27B）為每一題**寫 Python 程式**；程式必須在「偽裝過的題目」（旋轉＋換色的副本）上重現每一個訓練例，才會被採信。兩個通道的失敗方式不同：前者提出很多看似合理的網格，後者很少給答案、但給了通常是對的。最後用一條事先寫定的簡單規則，在「每題兩次機會」的計分方式下把兩者合併。

目錄：[1. 題目](#1-題目與難處) · [2. 管線](#2-管線總覽) · [3. 測試時訓練](#3-測試時訓練每題一個新-adapter) · [4. 解碼](#4-解碼列舉所有可能的網格) · [5. 增強重評分](#5-挑選候選跨對稱的一致性) · [6. 程式合成](#6-程式合成與驗證) · [7. 合併](#7-在-pass2-下合併兩個通道) · [8. 隱藏重跑的工程](#8-為-12-小時隱藏重跑做的工程) · [9. 量測方法](#9-排行榜有雜訊時怎麼量進步) · [10. 失敗的想法與原因](#10-沒有用的想法以及原因) · [11. 成績](#11-成績) · [12. 重現](#12-重現) · [術語表](#術語表)

### 1. 題目與難處
一個 ARC 題目給出幾組**輸入 → 輸出網格**（最大 30×30，每格是 0–9 的顏色），全部遵守同一條隱藏規則，另外給一個或多個測試輸入。解題者必須寫出完全正確的輸出網格：每一格和網格大小都要對。玩具例子（不是競賽資料）：

```
例 1：  1 0 0      0 0 1        規則：左右鏡射
        2 0 0  ->  0 0 2
        0 0 0      0 0 0
測試：  0 3 0  ->  ?           答案：0 3 0 / 4 0 0 / 0 0 0（鏡射）
        0 0 4
        0 0 0
```

ARC-AGI-2 對機器學習難在：
- **每條規則幾乎沒有資料。** 每題只有 2–5 個例子，隱藏測試題用的規則不會出現在訓練集裡。模型無法靠記憶，必須從眼前的例子推出規則。
- **完全比對計分。** 錯一格就是錯。每個測試輸出可以給兩次答案（**pass@2**），分數是解對的測試輸出比例。
- **嚴格的算力限制。** Kaggle 在隱藏題目上重跑 notebook，**不能連網、4 張 NVIDIA L4、總共 12 小時**。微調、搜尋、27B 模型全部要塞進這個預算。

### 2. 管線總覽
```
隱藏測試題
   │
   ├─ 第 1 階段：NVARC pass A（Qwen3-4B，4 張 GPU、每張一個 worker）     runs/arc-agi-2/v13/src/arc_solver.py
   │     每題：測試時 LoRA 訓練 → DFS 列舉 → 增強重評分 → 前兩名網格
   │     → 先寫出一份完整、合法的 submission.json                              （assemble.py）
   │
   ├─ 第 2 階段：程式合成（vLLM 上的 Qwen3.8-27B-FP8，4 卡張量平行）        v13_llm.py、progsynth.py
   │     每題、從最沒把握的開始：寫 transform(grid) → 在沙箱裡跑偽裝過的副本
   │     → 重現所有訓練對才採用
   │
   └─ 合併（事先登記）：驗證過的程式答案 → attempt_1，NVARC 第一名 → attempt_2     v13_merge.py
```
測試時訓練管線不是我們寫的。它來自 sorokin 公開的 **NVARC** 式 notebook（Apache-2.0），底層是 sorokin 在 ARC 類網格上預訓練的 Qwen3-4B（`sorokin/qwen3_4b_grids15_sft139`）。我們的貢獻是程式合成通道、合併規則、第 8 節的可靠性工程與第 9 節的量測方法；所有上游來源列在 `THIRD_PARTY.md`。

### 3. 測試時訓練：每題一個新 adapter
*程式：`runs/arc-agi-2/v13/src/arc_solver.py` 的 `solve_task()`、`worker()`；擴增在 `arc_loader.py`。*

**想法。** 在大量 ARC 類題目上預訓練過的模型，有一般性的先驗（物件、對稱、計數、顏色角色），但新題目的規則不在它的權重裡。測試時訓練（TTT）把規則放進去：回答前，先在**這一題自己的例子**上做幾步梯度下降。這裡過擬合就是目的，模型只需要對這一條規則正確。

**怎麼從 2–5 個例子變出訓練資料。** ARC 規則幾乎都不受以下變換影響：整題旋轉／翻轉、顏色重新命名、例子順序重排。每一種變換都產生一條新的、同樣有效的訓練序列（`ArcDataset.augment`）：
- 正方形的 8 種對稱（不變、3 種旋轉、轉置及其 3 種旋轉）：對題目的**每一張**網格套用 `transpose` × `rot90`；
- 10 種顏色的隨機置換（`permute_mod`），輸入與輸出一致套用；每個對稱副本各自獨立抽一次，所以 8 種對稱 × 8 次抽取＝64 條顏色各不相同的序列；
- 例子順序洗牌（`shuffle_ex`），也是每條序列各自抽。

一條訓練序列就是把整題寫成對話：每個例子是 `user: <輸入網格> / assistant: <輸出網格>`，網格寫成一列列數字（`QwenFormatter`）。**只在 assistant（輸出網格）的 token 上算 loss**（`QwenDataCollatorForCompletionOnlyLM` 把其他部分遮掉），所以模型學的是「給定這些例子和這個輸入，產生這個輸出」，而不是複製輸入。

**設定與理由。** LoRA rank 256，用 rank 穩定化縮放（rsLoRA），套在所有注意力與 MLP 投影層，以及嵌入層和輸出層；學習率 5e-5、cosine 排程、1 個 epoch、bf16。每題開始前把 adapter 重設回同一組初始權重（`set_peft_model_state_dict(... default_weights)`），題目之間不會互相洩漏。上游 notebook 每題用 16 種顏色置換 × 8 種對稱＝128 條序列；我們送出的 v13 用**半成本寬度** 8 × 8＝64 條（`arc_v6_config.py` 的 `N_TRAIN_AUG=8`），把 GPU 時間讓給第 2 階段。超過 8,192 token 的序列會刪掉例子直到放得下（`cut_to_len`）。

### 4. 解碼：列舉所有可能的網格
*程式：`arc_solver.py` 的 `turbo_dfs()`／`inference_turbo_dfs()`。*

TTT 之後，模型看測試輸入（同樣有好幾個對稱／換色「視角」），寫出輸出網格。解碼器不用取樣或 beam search，而是**對整個輸出空間做深度優先搜尋、依機率剪枝**：
- 詞表限制為 12 個 token：數字 0–9、換行、訊息結束。每一步分支這麼少，窮舉才負擔得起。
- 部分網格只有在累積負對數似然低於 `max_score = −ln 0.2` 時才繼續延伸。所以搜尋會回傳機率至少 20% 的完整網格，通常一到數個，其他一概不回傳。（前提是搜尋跑完：每次搜尋也會在 540 秒與該題截止時間被切斷。）
- `N_EVAL_AUG=1` 時，每個測試輸入用 **8 個視角**解碼（8 種對稱，各自一種顏色置換）；上游全寬是 16 個。
- 同一個測試輸入的各視角，以最多 4 個等長 prompt 為一批一起解碼（`arc_v6_config.py` 的 `decode_batches`）；每一步遞迴是一次批次前向計算，並沿用 KV cache。

為什麼不用取樣：完全比對計分獎勵的是最可能的那一張網格，而機率門檻能給出完整、確定性的候選清單，也不用調溫度。每個候選都會轉回原始方向與顏色（`invert_mod`）。

### 5. 挑選候選：跨對稱的一致性
*程式：`arc_solver.py` 的 `solve_task()`（重評分迴圈）；`arc_decoder.py` 的 `score_kgmon`。*

錯的答案可能碰巧在某一個視角裡很有把握。對的答案應該在**每一個**視角裡都仍然可能，因為規則是對稱的。所以每個不同的候選都會被重評分：把題目（以候選當測試輸出）擴增成 8 個新視角（全部 8 種對稱，各自有新的顏色置換與例子順序），在每個視角裡計算模型對整張候選網格的負對數似然——也就是所有 token 的**總和**（`calc_scores`）。再用 `kgmon` 分數排序：

```
kgmon(候選) =（解碼時產生這張網格的視角數）−（這張網格在 8 個重評分視角下的平均 NLL）
```

第一項獎勵各自獨立的解碼結果互相同意；第二項獎勵題目被偽裝後仍然可能的網格。前兩名成為 `attempt_1` 與 `attempt_2`。重評分結果依網格快取，重複出現的候選不會多花成本。

### 6. 程式合成與驗證
*程式：`runs/arc-agi-2/v13/src/v13_llm.py`（流程）、`progsynth.py`（prompt、沙箱、檢查）、`arc_llm.py`（擴增）。*

**想法。** 網格模型可能很接近但錯；**程式**則只有兩種結果：重現例子，或沒有。如果一支程式重現了每一組訓練對，它的測試輸出就比未驗證的猜測可能正確得多。驗證把一個低召回的產生器變成高精度的產生器。

**一個工作的步驟。**
1. **偽裝題目**（`arc_llm.augmentation`、`augment_task`）：一種確定性的非恆等對稱，加上顏色 1–9 的置換（0 維持背景）。這讓模型不容易認出看過的題目、背出記住的答案，也讓每次重複有不同視角。
2. **Prompt**（`progsynth.build_prompt`）：把例子和測試輸入寫成類 JSON 的列，要求在一個 Python 區塊裡給出完整的 `transform(grid)`。Qwen3.8-27B-FP8 開啟思考模式（temperature 1.0、top-p 0.95、top-k 20），由 vLLM 0.19.0 以 4 卡張量平行服務，搭配 MTP 推測解碼與 fp8 KV cache。
3. **收集候選**：最終回答的最後一個程式區塊，再加上推理過程中最後兩份程式草稿。草稿不多花 token，在長推理被 token 上限截斷時特別有用。
4. **在沙箱驗證**（`progsynth.run_program`、`check`）：每支程式一個獨立 Python 行程，限制記憶體（2 GB）、CPU 時間（20 秒）與檔案大小，工作目錄是空的暫存資料夾，stdout 與結果通道分開，逾時就殺掉整個行程群組。**驗證通過**＝每個訓練輸出都完全重現，且每個測試輸出都是合法網格。
5. **擋掉作弊程式**（`v13_llm.py` 的 `suspicious()`）：原始碼裡寫死訓練輸出（≥ 9 格、≥ 2 色）、測試答案（≥ 9 格）照抄某個訓練輸出、或沒有任何訓練對是恆等時測試答案卻等於測試輸入。
6. 把測試答案**轉回原座標與顏色**（`invert_aug`），並以原子方式重寫提交檔。

**排程。** 隱藏重跑時，NVARC 第一名 kgmon 分數低於 5.0（「沒把握」）的題目先排，越沒把握越前面；「有把握」的題目照檔案順序排在後面（NVARC 已經很有把握的題目，程式能加的不多）。並行數受 KV cache 記憶體限制：`int(kv_tokens / (0.35 × 上限 + 6000)) + 2` 條串流（`v13_llm.py`），64k 上限時是 13 條。沒拿到驗證程式的題目會換一個新偽裝排到隊尾，最多 6 次；同一題不會同時在跑兩個工作，後面的重複只給還沒解出的題目。

**實際效果，老實說。** 在 30 題公開評估題上做的 2 小時開發執行中，共開始 26 個工作：只有 2 個自然結束，11 個撞到 64k token 上限，13 個在時限到時被停止。4 支程式通過驗證，其中 3 支正確（4 支中有 2 支是從撞到上限的推理裡救回的草稿）。和 NVARC 合併後，三個重播種子都**多解了 2.0 題（共 30 題）**（種子 1 是 6.67 → 8.67；`artifacts/verification/arc-agi-2/v13/replay-v2-merge.json`）。數字很小，但看得出這個通道的特性：精度高、產量受推理長度綁住（兩個自然結束的工作各用了 47k 與 56k token）。這個限制是第 10 節多數實驗的動機。

**那次執行中的四個工作，從頭到尾**（題號來自公開評估集）：
- `dbff022c`：推理正常結束（47k token），程式通過驗證且正確；NVARC 沒有正確答案 → 程式答案佔 attempt_1：**多解 1 題**。
- `332f06d7`：推理撞到 64k 上限，但推理中的一份草稿通過驗證且正確 → **多解 1 題**。沒有擷取草稿的話，這個工作什麼都不會產生。
- `2b83f449`：同樣撞到上限；一份草稿通過所有訓練對，卻**在測試輸入上答錯**（偽陽性）。它擠掉了 NVARC 的 attempt_2，而那個答案本來也是錯的，所以這次沒損失——但這正是合併規則可能讓我們失分的情況。
- `53fb4810`：通過驗證；答案已經是 NVARC 兩個答案之一 → 該項不變。

### 7. 在 pass@2 下合併兩個通道
*程式：`runs/arc-agi-2/v13/src/v13_merge.py` 的 `fill_entry()`／`ranked_program_answers()`。*

每個測試輸出：最多驗證程式支持的答案（同票取最先驗證的）成為 `attempt_1`；NVARC 第一名成為 `attempt_2`。如果程式答案已經等於 NVARC 兩個答案之一，就不動。如果 NVARC 沒有產生真正的網格，`attempt_2` 用下一個不同的程式答案。

為什麼這樣排：有兩次機會時，放進程式答案唯一的代價是失去 NVARC 的**第二**選擇。驗證過的程式遠比 NVARC 的第二名常對，所以擠掉 attempt_2 通常代價很低——但不是零：偽陽性程式（如上面的 `2b83f449`）也會擠掉 attempt_2，若那個 attempt_2 本來是對的，這題就丟了。在重播裡，每個種子都有 3 個輸出被程式答案改動（種子 1、2：三個擠掉 attempt_2；種子 3：兩個擠掉 attempt_2、一個補上 NVARC 的空白項）；每個種子有 2 個因此變對，沒有任何原本正確的 NVARC 答案被擠掉（每一列 `lost_correct_nvarc: false`）。這條規則在量測前就寫定了。

### 8. 為 12 小時隱藏重跑做的工程
*程式：`runs/arc-agi-2/v13/src/assemble.py`、`arc_v6_config.py`、`arc_solver.py`（`worker`）、`v13_llm.py`。*

Kaggle 程式賽的 notebook 會在你看不到的資料上重跑，第 11 小時當掉就是零分。以下是各項防禦，以及它防止的失敗：
- **先寫完整的備援。** 任何模型載入前，就先有一份合法的 `submission.json`（每個測試輸出都是佔位網格）；NVARC 結果、之後每支驗證程式都以**原子方式**覆寫它（先寫入再 `os.replace`），任何時刻被殺掉都留下合法檔案。
- **隔離失敗。** 每張 GPU 一個行程；某題出例外就記錄後繼續下一題；連續 3 次失敗才讓 worker 退出，記憶體不足只算半次（即連續 6 次 OOM）；黏著型 CUDA 錯誤只會停掉那一個 worker。讀不到或寫到一半的結果會略過，不會致命。
- **管控總時間。** 全域截止時間（12 小時 − 10 分鐘）；每題解碼上限會縮成剩餘時間的公平份額（`adaptive_cap`），但不低於 360 秒，剩不到 360 秒就不開新題；測試時訓練本身不設時限，除非已過全域截止時間，每個測試輸入至少有一批解碼。
- **第 2 階段失敗時安全退回。** 第一個程式工作開始前的任何失敗（缺輸入、GPU 還被佔用、安裝、伺服器啟動）都會略過第 2 階段，NVARC 的提交原封不動；之後也只有驗證過的程式會動到檔案。（但它的**答案**仍可能透過合併規則讓我們失分，見第 7 節。）vLLM 在獨立的 Python 3.12 環境執行，不碰 NVARC 的套件；直譯器壓縮檔與 vLLM、torch 兩個 wheel 核對寫死在程式裡的 SHA-256，其餘 wheel 核對資料集的 `SHA256SUMS`。
- **測隱藏形狀。** 每次提交前，CPU 測試會在題目 ID、題數、順序都不同的測試檔上跑 notebook 邏輯，並注入當機與卡住（`tests/test_arc_agi_2_v6_hidden_shape.py`、`runs/arc-agi-2/v13/tests*/`）。

### 9. 排行榜有雜訊時怎麼量進步
- **公開分是有雜訊的樣本。** 同一系列換種子重交，公開分的標準差約 0.9 分；我們把 **2.4 分**以內的差距當雜訊。公開榜也只涵蓋隱藏集的一部分。用最高公開分挑最終提交，挑到的多半是運氣。
- **固定的離線題組。** 我們在 30 題公開評估題（「keys30」，`keys30.json`）上量測，事先登記重複次數與種子的組合，回報 30 題中解出幾題。
- **貴的階段快取、便宜的階段重播。** keys30 的 NVARC 輸出在單 GPU VM 上每個種子各跑一次（實驗 E1）並存檔；之後任何新的合併規則或程式集合，都能零 GPU 在上面計分（`runs/arc-agi-2/v13/offline_v13.py`、`replay_*.py`）。
- **依容量調整。** 隱藏重跑中，同樣的 LLM 時間要分給比 keys30 多好幾倍的題目，不是每題都能得到離線時的嘗試次數。所以程式通道相對「只有 NVARC」的全部增益，要乘上事先登記的容量係數 c：`調整後 = HA + c × (原始 − HA)`，HA＝6.778 是半寬 NVARC 單獨的分數，64k token 上限時 c＝0.65、96k 時 0.37（`prereg-e6-v4-seeds-v1.md`、`c2_measure.py`）。本 README 出現的幾個數字：

  | 量 | v13 v2 |
  |---|---:|
  | HA：半寬 NVARC 單獨（E1，3 個種子平均） | 6.778 |
  | 單次開發執行與 HA 合併（`v2_dev`） | 8.778 |
  | 原始：事先登記重複的平均（E3） | 8.278 |
  | **調整後**：6.778 + 0.65 × (8.278 − 6.778) | **7.753** |
- **事先登記。** 每次較大的執行前，我們先寫下指標、判定規則與停止規則（`artifacts/verification/arc-agi-2/v13/prereg-*.md`），結果出來後不能再重新解讀。

### 10. 沒有用的想法，以及原因
| 想法 | 預期 | 實際 | 原因（我們的解讀） |
|---|---|---|---|
| token 上限 64k → 96k（v4／E4、E6） | 更多推理能結束、更多程式通過驗證（產量模型預測公開約 +2.3 分；`cap-yield-v1.json`） | 未調整的離線增益（9.64 對 8.28）在容量調整後消失（7.84 對 7.75，打平）；單次公開讀數 **−3.19** | 工作變長，時間窗內能跑的工作變少，KV 記憶體裡的平行槽位也變少（13 → 10 條）。在隱藏集的大量題目上，能得到嘗試的題目變少；容量係數從 0.65 掉到 0.37。預登記也註明：公開分的下跌無法區分是覆蓋率下降，還是偽陽性程式擠掉了正確的 NVARC 答案。 |
| 強制縮短推理（budget forcing，v3） | 思考變短 → 工作變多 | 開發重播 7.11 對 8.78 | 難的 ARC 規則需要長時間探索；砍掉它失去的解比多跑的工作找到的多。 |
| 「medium」推理強度＋一輪執行回饋修補（C2） | 推理變短，並修好差一點的程式 | 比 v2 少 0.98 題（調整後 6.778 對 7.753）；P(Δ>0)＝0 | 記錄的列數變多（相近時間內 83 對 26，含修補輪次），但通過驗證的程式變少（3 對 4）。我們的推測——較低強度寫出的程式較不嚴謹、修補輪次把預算花在更多推理上——沒有用另外的實驗分離驗證。 |
| 程式通道加入 ARC-1 式 DSL 搜尋（C1） | 便宜的額外解 | 重播中零貢獻（7.753＝v2） | 它能解的題目，很可能 NVARC 或 LLM 已經解出。 |
| 一次全寬 NVARC 對兩次半寬合併（E1、v8／v11） | 第二輪帶來多樣性 | 事先登記的判定為「無法判斷」 | n＝30 時差距在種子雜訊內。 |
| 網格模型基線（Qwen2.5-Coder-7B、Qwen3-2B） | — | 公開 0.00／4.58 | 沒有 ARC 專用預訓練與測試時訓練，模型解不出隱藏題。 |

### 11. 成績
公開排行榜（pass@2，隱藏集的公開部分）。本 repo 不記錄私榜分數。

| 提交 | 內容 | 公開 |
|---|---|---:|
| 57165541 | v13 v2：NVARC pass A＋驗證式程式合成——**最終選定 1** | 32.22 |
| 56864814 | v8：NVARC pass A＋換種子的 pass B 合併——**最終選定 2**（不同系列，當避險） | 30.97 |
| 57828532 | v13 v5（同程式、新種子） | 33.89（在雜訊帶內；與選定 1 同系列，未選） |
| 其他 NVARC 變體與公開 fork | v6、v7、v9、v11、perfpatch、Boyko V22 | 28.89–30.56 |

2026-11-02（約 18:40 UTC，3,627 隊）公開榜銅牌線是 32.22。完整提交歷史在 `artifacts/verification/arc-agi-2/`。

### 12. 重現
- **在 Kaggle 上。** Notebook 用 4× NVIDIA L4、關閉網路、12 小時。掛載競賽、`sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`、`mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1`、資料集 `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`、kernel `sorokin/pip-install-unsloth-flash-patch`，以及我們的 Python 3.12 下載 kernel https://www.kaggle.com/code/f0909172434/arc-2026-py312-fetch-v1（原始碼在 `runs/arc-agi-2/v13/fetch/`）。送出的 notebook 在 `runs/arc-agi-2/v13/upload-arc-2026-v13-progsynth-v2/`：把 `kernel-metadata.json` 的 `id` 改成你的帳號，`kaggle kernels push -p <資料夾>`，再提交那個版本。commit 執行（沒有 `KAGGLE_IS_COMPETITION_RERUN`）是開發量測，跑 4 題冒煙測試＋keys30；競賽重跑則在隱藏題上跑完整流程。v8 在 `runs/arc-agi-2/v8/upload-arc-2026-v8-4b-v2/`。
- **重建。** `python runs/arc-agi-2/v13/build.py <slug>`（需要 `nbformat`）。目前的建置器會多打包送出後才加入的 `model_profiles.py`，所以重建結果功能相同，但與送出的 notebook（SHA-256 `f93be126…`）不會逐位元組相同。
- **CPU 測試。** `pip install -r requirements.txt`，再跑 `python -m pytest runs/arc-agi-2/v13/tests` 與 `tests5`；`tests/` 的檔案請逐一執行（v8／v9／v11 的測試以相同模組名稱載入不同的 `src/`）。在我們的乾淨副本上全部通過，只有一項需要未公開發射層的測試被略過，以及 `tests/test_arc_agi_2_fork_audit.py::test_all_log_files_concatenated` 一項原本就失敗。重播需要 E1 解碼輸出，本 repo 未收錄。
- **資料。** 本 repo 不含 ARC 題目檔、Kaggle 檔案或模型權重。請從競賽頁或 https://github.com/arcprize/ARC-AGI-2 取得題目，放在 `data/arc-agi-2/official-v1/`。

### Repo 結構
```
runs/arc-agi-2/v13/src/        送出的 v13 v2 原始碼（NVARC 檔案＋v13_llm.py、progsynth.py、arc_llm.py、v13_merge.py）
runs/arc-agi-2/v13/            建置器、稽核、離線重播、消融實驗原始碼（src3、src5、src6、src7、arc1/）
runs/arc-agi-2/v6 … v11/       NVARC 系列（v8＝最終選定 2；v6/author-v10＝上游 cell）
runs/arc-agi-2/gcp-e1/         離線量測用的單 GPU 重播工具
artifacts/verification/arc-agi-2/   預登記、稽核、隱藏形狀結果、重播結果
tests/                         隱藏形狀與 pass B 測試
```

<a id="術語表"></a>
### 術語表：專案怎麼運作
本專案由 AI 程式代理（Claude Code 對話）在人類監督下執行：人類擁有者設定目標、核准花費與公開動作，並對有風險的步驟做最後決定；代理寫程式、跑實驗、互相審查。紀錄中保留了這些用語：

| 用語 | 意思 |
|---|---|
| owner session／協調者 | 負責本競賽的對話／在各競賽間分配 GPU 額度、轉達公開 notebook 掃描結果的協調對話 |
| Argon、Opus、Fable、advisor | 獨立審查模型（Gemini 4 Argon；經 API 的 Claude Opus；最終選定用的 Claude Fable；重要決定前讀完整對話紀錄的較強模型） |
| T0–T3、P0–P2 | 審查分級（不審／一份審查／大額 GPU 或共用規則用兩份盲審／最終選定用完整閘門）與發現嚴重度（P0＝必修） |
| R1、R4 | 內部規則：探索性提交只需自動閘門；最終選定依事先登記的離線估計，只有公開分的候選必須比有離線支持的最佳版本高出 2 倍雜訊帶以上 |
| 預登記（`prereg-*.md`） | 執行前寫下的指標、判定規則與停止規則 |
| 隱藏形狀測試 | 確認 notebook 在題目 ID、題數、順序都不同時也能正常執行的測試 |
| keys30、HA、HB、F、E1–E6、C1–C3 | 30 題開發題；半成本 pass A／pass B；全寬單輪；編號的實驗 |

### 授權與署名
Apache License 2.0，見 `LICENSE` 與 `NOTICE`。NVARC 流程檔案衍生自 sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue（Apache-2.0）；所有上游程式、模型、資料集與想法都列在 `THIRD_PARTY.md`。
