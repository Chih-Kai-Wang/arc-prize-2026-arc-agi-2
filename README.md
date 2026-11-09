# Solving ARC-AGI-2 puzzles with test-time training and self-checking programs

**ARC Prize 2026 · Kaggle code competition · code and write-up**

[English](#english) · [繁體中文](#繁體中文)

---

<a id="english"></a>

## Abstract

ARC-AGI-2 is a set of visual reasoning puzzles. Each puzzle shows a few examples of a hidden rule, and the solver must apply the rule to a new grid exactly. This document explains, for readers without a machine-learning background, how our Kaggle solution works and why each part helps.

The solution has two independent channels. **Channel A** briefly retrains a small language model on each puzzle and lists the answers it finds most likely. **Channel B** asks a larger model to write a short computer program for the puzzle, and accepts the program only if it reproduces every example. A simple rule, fixed in advance, combines the two. Our main submission scored 32.22 on the public leaderboard.

## Contents

1. [Introduction](#1-introduction)
2. [Background](#2-background)
3. [Channel A: learning each puzzle at test time](#3-channel-a-learning-each-puzzle-at-test-time)
4. [Channel B: programs that check themselves](#4-channel-b-programs-that-check-themselves)
5. [Combining the two channels](#5-combining-the-two-channels)
6. [Running safely on hidden data](#6-running-safely-on-hidden-data)
7. [Measuring progress](#7-measuring-progress)
8. [What did not work](#8-what-did-not-work)
9. [Results](#9-results)
- [Appendix A. Reproducing the submission](#appendix-a-reproducing-the-submission)
- [Appendix B. Where each part lives in the code](#appendix-b-where-each-part-lives-in-the-code)
- [Appendix C. How the project was run](#appendix-c-how-the-project-was-run)

---

## 1. Introduction

### 1.1 The task

> **Definition 1.1 (puzzle).** A *grid* is a rectangle of at most 30 × 30 cells, each coloured with one of 10 colours, written 0–9. A *puzzle* consists of 2 to 5 *example pairs* (input grid → output grid) that all follow one hidden rule, and one or more *test inputs*.

The solver must produce the output grid for each test input. The answer counts only if every cell and the size of the grid are correct.

> **Example 1.2.** A toy puzzle (not from the competition) whose rule is "mirror left to right":
>
> ```
> example:   1 0 0      0 0 1          test:   0 3 0      0 3 0
>            2 0 0  →   0 0 2                  0 0 4  →   4 0 0
>            0 0 0      0 0 0                  0 0 0      0 0 0
> ```

> **Definition 1.3 (score).** Two attempts are allowed for each test output. The *score* is the percentage of test outputs for which one of the two attempts is exactly correct. This is often written *pass@2*.

### 1.2 Why it is hard

1. **Very little data per rule.** Each rule is shown only 2–5 times, and the hidden puzzles use rules that are not in any training set. A system cannot memorise answers; it must infer the rule from the examples in front of it.
2. **All or nothing.** One wrong cell makes the whole answer wrong.
3. **Fixed computing budget.** Kaggle runs our notebook on the hidden puzzles with no internet access, four GPUs (NVIDIA L4) and 12 hours in total.

### 1.3 Our approach

```
              hidden puzzles
                    │
     ┌──────────────┴──────────────┐
     │                             │
 Channel A                     Channel B
 retrain a small model         a large model writes a program;
 on each puzzle, list          keep it only if it reproduces
 the most likely answers       every example pair
 (many answers, often close)   (few answers, usually right)
     │                             │
     └──────────────┬──────────────┘
                    │
        combine with a fixed rule (§5)
                    │
              two attempts per test output
```

The two channels make different kinds of mistakes, so together they solve more puzzles than either alone.

Channel A is not our invention. It is the public "NVARC" notebook by Kaggle user sorokin (Apache-2.0 licence), which uses a small model that sorokin trained on ARC-like grids. Our contributions are:

- Channel B and the rule that combines the two channels;
- the safeguards that keep a 12-hour hidden run from failing (§6);
- the way we measured progress on a noisy leaderboard (§7).

---

## 2. Background

This section defines the few machine-learning terms the rest of the document uses.

> **Definition 2.1 (language model, token).** A *language model* reads a sequence of symbols, called *tokens*, and gives a probability for each possible next token. Writing text means choosing tokens one at a time. In this project a grid is written as rows of digits, one row per line, so each cell is one token.

> **Definition 2.2 (probability of an answer).** The probability that the model writes a whole grid is the product of the probabilities of its tokens. It is convenient to use the *negative log-likelihood*
> $$\mathrm{NLL}(\text{grid}) = -\sum_{\text{tokens}} \log p(\text{token}) \tag{2.1}$$
> A smaller NLL means the model finds the grid more likely. NLL = 0 means certainty.

> **Definition 2.3 (fine-tuning, LoRA).** *Fine-tuning* adjusts a trained model's parameters on new examples so that it imitates them. *LoRA* is a cheap form of fine-tuning: the original parameters stay fixed, and a small set of extra parameters, called an *adapter*, is learned and added on top. An adapter can be thrown away and re-created quickly.

> **Definition 2.4 (augmentation).** An *augmentation* is a change to a puzzle that keeps its rule valid. In ARC the main ones are rotating or reflecting every grid, renaming the colours, and reordering the example pairs.

---

## 3. Channel A: learning each puzzle at test time

The model has seen many ARC-like puzzles before the competition, so it knows general ideas such as objects, symmetry and counting. It has not seen the rule of a new puzzle. Channel A therefore teaches it that rule just before answering. This is called *test-time training*.

### 3.1 Making training data from a few examples

A square has 8 symmetries: 4 rotations, each with or without a reflection. Applying one of them to every grid of a puzzle gives a new puzzle with the same rule. So does renaming the colours, and so does reordering the example pairs.

For each puzzle, Channel A creates
$$8 \text{ symmetries} \times 8 \text{ colour renamings} = 64 \text{ versions} \tag{3.1}$$
each with its own random order of the example pairs. The original notebook uses 16 renamings (128 versions); we used 64 to save time for Channel B.

### 3.2 What the model learns

Each version is written as a conversation: the model is shown an input grid and must write the matching output grid, for each example pair in turn. A fresh LoRA adapter (Definition 2.3) is trained on the 64 versions for one pass.

Two details matter:
- The model is graded only on the **output** grids it writes, not on the inputs. It learns "given these examples and this input, produce this output".
- Before each new puzzle the adapter is reset. What the model learned for one puzzle never leaks into another.

Over-fitting, normally a problem in machine learning, is the goal here: the model only has to be right about one rule.

### 3.3 Listing the likely answers

The trained model now reads the test input and writes an output grid. The test input is also shown in 8 versions (the 8 symmetries, each with its own colour renaming). Each version is called a *view*.

Instead of writing just one answer, the model searches for **every grid whose probability is at least 20%**, that is,
$$\mathrm{NLL}(\text{grid}) < -\log 0.2 \approx 1.61. \tag{3.2}$$

This search is affordable because the model may only write 12 different tokens (the digits 0–9, "new line" and "end"). Any partial grid whose NLL already exceeds the limit is abandoned. Usually one to a few grids survive. Each surviving grid is rotated and recoloured back to the original puzzle's orientation.

> **Remark 3.1.** The search also stops after a time limit, so in rare cases it may not find every grid above 20%.

### 3.4 Choosing two answers

A wrong answer can look likely in one view by accident. A correct answer should look likely in **every** view, because the rule does not depend on orientation or colour names. Channel A uses this to rank the candidates.

For each candidate grid $g$:
1. count $n(g)$, the number of views in which the search produced $g$;
2. disguise the puzzle in 8 new ways, put $g$ in as the answer, and compute its NLL (2.1) in each;
3. combine the two into one score:
$$\mathrm{score}(g) = n(g) - \frac{1}{8}\sum_{k=1}^{8} \mathrm{NLL}_k(g). \tag{3.3}$$

The two highest-scoring grids become the two attempts.

---

## 4. Channel B: programs that check themselves

### 4.1 The idea

A model that writes grids directly can be close but wrong. A *program* behaves differently: run it on the example inputs, and it either reproduces the example outputs or it does not. If it reproduces all of them, its answer on the test input is much more likely to be correct.

We call such a program *verified*.

### 4.2 Procedure

For each puzzle, Channel B repeats the following steps.

1. **Disguise the puzzle.** Apply one symmetry and one colour renaming (Definition 2.4). This prevents the model from recognising a puzzle it may have seen and simply recalling the answer.
2. **Ask for a program.** A large model (Qwen3.8, 27 billion parameters) receives the example pairs and test inputs as lists of numbers and is asked to write a Python function `transform(grid)`. The model reasons in text before writing code. A reasoning trace may run to tens of thousands of tokens.
3. **Collect candidate programs.** Take the program in the final answer, plus the last two drafts written during the reasoning. Drafts are useful when the reasoning is cut off by the length limit.
4. **Run each program in a sandbox.** The sandbox is an isolated process with limits on memory and time, so a faulty program cannot harm the rest of the system.
5. **Verify.** Accept the program only if it reproduces **every** example output exactly and returns a valid grid for every test input.
6. **Reject obvious tricks.** Refuse programs that copy an example output into the code, or that return an example output or the unchanged test input as the test answer.
7. **Undo the disguise** on the test answer, and save it.

Puzzles that Channel A is least sure about are handled first. A puzzle with no verified program is tried again later with a new disguise, up to 6 times.

### 4.3 What happened in practice

In a 2-hour trial on 30 public puzzles, Channel B started 26 attempts. Only 2 finished their reasoning; 11 were cut off by the length limit (64,000 tokens) and 13 by the time limit. Four programs were verified, and three of them were correct.

Table 4.1 follows four of these attempts from start to finish.

| Puzzle | What happened | Effect on the score |
|---|---|---|
| `dbff022c` | Reasoning finished; the program was verified and correct. Channel A had no correct answer. | **+1 puzzle** |
| `332f06d7` | Reasoning was cut off, but a draft found inside it was verified and correct. | **+1 puzzle** |
| `2b83f449` | Cut off; a draft reproduced every example but gave a **wrong** test answer. | none here, but see Remark 5.2 |
| `53fb4810` | Verified; its answer was already one of Channel A's two attempts. | none |

**Table 4.1.** Four Channel B attempts from the trial run.

Combined with Channel A, Channel B added **2 solved puzzles out of 30**, in each of three repeated trials. The main limitation is clear from the table: answers are reliable, but long reasoning means few attempts finish.

---

## 5. Combining the two channels

> **Rule 5.1 (combination).** For each test output:
> 1. If at least one verified program exists, its answer becomes **attempt 1** and Channel A's best answer becomes **attempt 2**. (If several programs disagree, the answer given by the most programs is used.)
> 2. If the program's answer is already one of Channel A's two attempts, nothing changes.
> 3. If there is no verified program, Channel A's two answers are kept.

> **Remark 5.2 (cost).** Rule 5.1 replaces Channel A's *second* attempt. This costs a point only when a verified program is wrong **and** Channel A's second attempt was right. Verified programs are rarely wrong, so the cost is small but not zero; puzzle `2b83f449` in Table 4.1 is an example of a wrong verified program. In our trial, no correct Channel A answer was lost.

Rule 5.1 was written down before the trial was run, so the result could not influence the rule.

---

## 6. Running safely on hidden data

Kaggle runs the notebook on puzzles we never see. A crash in the eleventh hour scores zero. Table 6.1 lists each risk and the safeguard against it.

| Risk | Safeguard |
|---|---|
| The run is stopped before it ends. | A complete answer file with placeholder answers is written first. Every improvement replaces the file in a single step, so a valid file always exists. |
| One puzzle causes an error. | The error is recorded and the next puzzle starts. Each GPU runs its own process, so one failing process does not stop the others. |
| Time runs out. | A global deadline, and a per-puzzle time limit that shrinks as the deadline approaches. |
| Channel B fails to start. | If anything goes wrong before Channel B's first attempt, Channel B is skipped and Channel A's answers are kept. |
| The hidden puzzles differ from the public ones. | Before each submission, tests run the notebook on files with different puzzle names, counts and order, with simulated crashes. |

**Table 6.1.** Risks of the hidden run and the corresponding safeguards.

---

## 7. Measuring progress

### 7.1 The leaderboard is noisy

Submitting the same method twice with different random seeds changed the public score by about 0.9 points (one standard deviation). We therefore treated differences smaller than **2.4 points** as noise. Choosing a final submission by the highest public score would often choose luck.

### 7.2 A fixed practice set

We measured every idea on the same 30 public puzzles, called *keys30*, and repeated each measurement with several seeds. Expensive steps were run once and saved; cheaper later steps were then replayed on the saved results at no GPU cost.

### 7.3 Correcting for the larger hidden set

The hidden set contains several times more puzzles than keys30, but Channel B has the same 12-hour budget. Each puzzle therefore gets fewer attempts than in practice. We corrected for this with a factor $c$, fixed in advance:
$$\text{adjusted} = A + c\,(\text{measured} - A), \tag{7.1}$$
where $A$ is the score of Channel A alone. For our submitted version, $A = 6.778$, measured $= 8.278$ and $c = 0.65$, so

$$\text{adjusted} = 6.778 + 0.65 \times (8.278 - 6.778) = 7.753 \text{ puzzles out of 30}.$$

### 7.4 Deciding in advance

Before each larger experiment we wrote down what we would measure, which result would count as success, and when we would stop. This prevents reinterpreting a result after seeing it.

---

## 8. What did not work

| Idea | Result | Our explanation |
|---|---|---|
| Let Channel B reason longer (limit 96k instead of 64k tokens) | Practice score unchanged after the correction (7.84 vs 7.75); one public reading 3.19 points lower | Longer attempts mean fewer attempts and fewer puzzles reached in 12 hours; the factor $c$ in (7.1) fell from 0.65 to 0.37. We could not separate this from wrong programs replacing correct answers. |
| Force Channel B to reason less | 7.11 vs 8.78 in practice | Hard rules need long reasoning; cutting it lost more than it gained. |
| A lower "reasoning effort" setting plus one round of feedback | about 1 puzzle worse | More attempts were logged but fewer programs were verified. The exact cause was not isolated. |
| A hand-written search over simple grid operations | no change | Most likely it solved only puzzles the other channels already solved. |
| Two shorter runs of Channel A instead of one long run | inconclusive | The difference was within the noise of 30 puzzles. |
| Smaller grid models without test-time training | public 0.00 and 4.58 | Without test-time training they could not handle new rules. |

**Table 8.1.** Ideas we tested and rejected.

---

## 9. Results

| Submission | Description | Public score |
|---|---|---:|
| 57165541 | Channel A + Channel B — **final choice 1** | 32.22 |
| 56864814 | Channel A run twice with different seeds — **final choice 2** | 30.97 |
| 57828532 | Same code as final choice 1, new seeds and a fixed concurrency setting | 33.89 |

**Table 9.1.** Main submissions. The third row differs from the first by less than the noise level of §7.1, so it was not chosen.

On 2 November 2026 (about 18:40 UTC, 3,627 teams) the public bronze-medal line was 32.22. Private-leaderboard scores are not recorded in this repository.

---

## Appendix A. Reproducing the submission

**On Kaggle.**
1. Create a notebook with 4 × NVIDIA L4 GPUs and internet turned off.
2. Attach the competition data and these inputs:
   - model `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`;
   - model `mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1`;
   - dataset `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`;
   - notebook `sorokin/pip-install-unsloth-flash-patch`;
   - our Python 3.12 download notebook https://www.kaggle.com/code/f0909172434/arc-2026-py312-fetch-v1 (source in `runs/arc-agi-2/v13/fetch/`).
3. In `runs/arc-agi-2/v13/upload-arc-2026-v13-progsynth-v2/kernel-metadata.json`, change `id` to your user name.
4. Run `kaggle kernels push -p runs/arc-agi-2/v13/upload-arc-2026-v13-progsynth-v2` and submit that version.

Final choice 2 is in `runs/arc-agi-2/v8/upload-arc-2026-v8-4b-v2/`.

**Tests on a CPU.**
```
pip install -r requirements.txt
python -m pytest runs/arc-agi-2/v13/tests
```
Run the files in `tests/` one at a time. All pass except one test that needs our unpublished launch tools (skipped) and one known failure in `tests/test_arc_agi_2_fork_audit.py`.

**Data.** This repository contains no puzzle files and no model weights. Download the puzzles from the competition page or https://github.com/arcprize/ARC-AGI-2 into `data/arc-agi-2/official-v1/`.

## Appendix B. Where each part lives in the code

| Section | File (under `runs/arc-agi-2/v13/src/`) | Main functions |
|---|---|---|
| §3.1 augmentations | `arc_loader.py` | `ArcDataset.augment` |
| §3.2 training | `arc_solver.py` | `solve_task`, `worker` |
| §3.3 search | `arc_solver.py` | `turbo_dfs` |
| §3.4 ranking | `arc_solver.py`, `arc_decoder.py` | `calc_scores`, `score_kgmon` |
| §4 Channel B | `v13_llm.py`, `progsynth.py`, `arc_llm.py` | `run_job`, `check`, `suspicious` |
| §5 Rule 5.1 | `v13_merge.py` | `fill_entry` |
| §6 safeguards | `assemble.py`, `arc_v6_config.py` | `adaptive_cap` |
| §7 measurement | `../offline_v13.py`, `../c2_measure.py` | — |

**Technical settings, for readers who want them.** LoRA rank 256 with rank-stabilised scaling, on all attention and feed-forward layers plus the embedding and output layers; learning rate 5 × 10⁻⁵, cosine schedule, one epoch, bf16. Channel B uses Qwen3.8-27B-FP8 served by vLLM 0.19.0 on four GPUs, sampling temperature 1.0, top-p 0.95, top-k 20; the sandbox limits each program to 2 GB of memory and 20 s of CPU time. Pre-registrations and measurement records are in `artifacts/verification/arc-agi-2/v13/`.

## Appendix C. How the project was run

The project was carried out by AI coding agents (Claude Code sessions) under the supervision of the human owner, who set the goals, approved spending and public actions, and made the final decisions. The records use some internal terms:

| Term | Meaning |
|---|---|
| owner session, orchestrator (協調者) | the session in charge of this competition; a session that shared computing time across competitions |
| Argon, Opus, Fable, advisor | AI models used as independent reviewers |
| T1–T3, P0–P2 | review levels and the severity of review findings |
| pre-registration (`prereg-*.md`) | a plan written before an experiment (§7.4) |
| keys30, E1–E6, C1–C3 | the practice set (§7.2); numbered experiments |

## Licence

Apache License 2.0; see `LICENSE` and `NOTICE`. Channel A derives from sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue (Apache-2.0). All upstream code, models and data are listed in `THIRD_PARTY.md`.

---

<a id="繁體中文"></a>

# 用測試時訓練與自我檢查的程式解 ARC-AGI-2 謎題

**ARC Prize 2026 · Kaggle 程式競賽 · 程式碼與說明**

## 摘要

ARC-AGI-2 是一組視覺推理謎題。每一題給出隱藏規則的幾個例子，解題者必須把規則準確套用到新的網格上。本文寫給沒有機器學習背景的讀者，說明我們的 Kaggle 解法如何運作、每個部分為什麼有幫助。

解法有兩個獨立的通道。**通道 A** 針對每一題短暫重新訓練一個小型語言模型，列出它認為最可能的答案。**通道 B** 請一個較大的模型為這一題寫一小段程式，只有在程式重現所有例子時才採用。最後用一條事先寫定的簡單規則合併兩者。我們的主要提交在公開排行榜得到 32.22 分。

## 目錄

1. [前言](#1-前言)
2. [背景知識](#2-背景知識)
3. [通道 A：測試時學會每一題](#3-通道-a測試時學會每一題)
4. [通道 B：會自我檢查的程式](#4-通道-b會自我檢查的程式)
5. [合併兩個通道](#5-合併兩個通道)
6. [在看不到的資料上安全執行](#6-在看不到的資料上安全執行)
7. [如何量測進步](#7-如何量測進步)
8. [沒有用的想法](#8-沒有用的想法)
9. [成績](#9-成績)
- [附錄 A　重現提交](#附錄-a重現提交)
- [附錄 B　各部分在程式碼中的位置](#附錄-b各部分在程式碼中的位置)
- [附錄 C　專案的運作方式](#附錄-c專案的運作方式)

---

## 1. 前言

### 1.1 題目

> **定義 1.1（謎題）。** *網格*是最多 30 × 30 格的長方形，每一格塗上 10 種顏色之一，記為 0–9。一道*謎題*包含 2 到 5 組*例子*（輸入網格 → 輸出網格），全部遵守同一條隱藏規則，另外還有一個或多個*測試輸入*。

解題者必須為每個測試輸入寫出輸出網格。只有每一格和網格大小都正確，答案才算對。

> **例 1.2。** 一道規則為「左右鏡射」的玩具謎題（不是競賽題目）：
>
> ```
> 例子：  1 0 0      0 0 1          測試：  0 3 0      0 3 0
>         2 0 0  →   0 0 2                  0 0 4  →   4 0 0
>         0 0 0      0 0 0                  0 0 0      0 0 0
> ```

> **定義 1.3（分數）。** 每個測試輸出可以作答兩次。*分數*是兩次作答中有一次完全正確的測試輸出所佔的百分比，常寫作 *pass@2*。

### 1.2 為什麼難

1. **每條規則的資料極少。** 每條規則只出現 2–5 次，而隱藏謎題的規則不在任何訓練資料裡。系統無法背答案，必須從眼前的例子推出規則。
2. **全對或全錯。** 錯一格，整個答案就錯。
3. **固定的計算預算。** Kaggle 在隱藏謎題上執行我們的 notebook，不能連網，只有 4 張 GPU（NVIDIA L4），總共 12 小時。

### 1.3 我們的做法

```
                 隱藏謎題
                    │
     ┌──────────────┴──────────────┐
     │                             │
   通道 A                        通道 B
 針對每一題重新訓練            大模型寫一段程式；
 小模型，列出最可能的答案      重現所有例子才採用
 （答案多，常常很接近）        （答案少，通常是對的）
     │                             │
     └──────────────┬──────────────┘
                    │
         用固定規則合併（第 5 節）
                    │
          每個測試輸出兩次作答
```

兩個通道犯的錯不同，合起來能解出比任一個單獨更多的題目。

通道 A 不是我們發明的。它是 Kaggle 使用者 sorokin 公開的「NVARC」notebook（Apache-2.0 授權），用的是 sorokin 在 ARC 類網格上訓練的小模型。我們的貢獻是：

- 通道 B，以及合併兩個通道的規則；
- 讓 12 小時隱藏執行不會失敗的防護措施（第 6 節）；
- 在有雜訊的排行榜上量測進步的方法（第 7 節）。

---

## 2. 背景知識

本節定義後文會用到的幾個機器學習名詞。

> **定義 2.1（語言模型、token）。** *語言模型*讀入一串符號（稱為 *token*），並對下一個可能的 token 給出機率。寫文字就是一次選一個 token。在本專案中，網格寫成一列列數字、每列一行，所以每一格就是一個 token。

> **定義 2.2（答案的機率）。** 模型寫出整張網格的機率，是其中每個 token 機率的乘積。為了方便，我們用*負對數似然*
> $$\mathrm{NLL}(\text{網格}) = -\sum_{\text{token}} \log p(\text{token}) \tag{2.1}$$
> NLL 越小，表示模型認為這張網格越可能；NLL = 0 表示完全確定。

> **定義 2.3（微調、LoRA）。** *微調*是用新的例子調整已訓練好的模型參數，讓它模仿這些例子。*LoRA* 是一種便宜的微調：原本的參數不動，只學一小組額外參數（稱為 *adapter*）疊加上去。adapter 可以很快丟掉、重新建立。

> **定義 2.4（擴增）。** *擴增*是對謎題做的、不破壞規則的改變。在 ARC 中主要有：旋轉或翻轉每一張網格、重新命名顏色、調換例子的順序。

---

## 3. 通道 A：測試時學會每一題

模型在比賽前已看過很多 ARC 類謎題，所以懂得物件、對稱、計數這類一般概念，但沒看過新謎題的規則。所以通道 A 在作答前先教它這條規則，這稱為*測試時訓練*。

### 3.1 從少數例子做出訓練資料

正方形有 8 種對稱：4 種旋轉，各自可以翻轉或不翻轉。對謎題的每一張網格套用其中一種，就得到規則相同的新謎題。重新命名顏色、調換例子順序也一樣。

通道 A 對每一題產生
$$8 \text{ 種對稱} \times 8 \text{ 種顏色命名} = 64 \text{ 個版本} \tag{3.1}$$
每個版本的例子順序也各自隨機排列。原 notebook 用 16 種顏色命名（128 個版本）；我們用 64 個，把時間留給通道 B。

### 3.2 模型學到什麼

每個版本寫成一段對話：模型看到一張輸入網格，必須寫出對應的輸出網格，依序對每組例子進行。在這 64 個版本上，從頭訓練一個新的 LoRA adapter（定義 2.3），跑一輪。

有兩個重要細節：
- 模型只按它寫出的**輸出**網格評分，不按輸入。它學的是「給定這些例子和這個輸入，寫出這個輸出」。
- 每一題開始前，adapter 都重設。從一題學到的東西不會帶到另一題。

過擬合在機器學習中通常是問題，在這裡卻是目的：模型只需要對這一條規則正確。

### 3.3 列出可能的答案

訓練好的模型接著讀測試輸入、寫出輸出網格。測試輸入同樣以 8 個版本呈現（8 種對稱，各自一種顏色命名），每個版本稱為一個*視角*。

模型不是只寫一個答案，而是搜尋**所有機率至少 20% 的網格**，也就是
$$\mathrm{NLL}(\text{網格}) < -\log 0.2 \approx 1.61。 \tag{3.2}$$

這個搜尋負擔得起，因為模型只能寫 12 種 token（數字 0–9、「換行」、「結束」）。只要寫到一半的網格 NLL 已超過上限，就放棄這一支。通常會留下一到數張網格。每張留下的網格再轉回原謎題的方向與顏色。

> **註 3.1。** 搜尋也有時間上限，所以少數情況下可能沒找齊所有機率超過 20% 的網格。

### 3.4 選出兩個答案

錯的答案可能碰巧在某個視角裡看起來很可能。對的答案應該在**每一個**視角裡都很可能，因為規則和方向、顏色名稱無關。通道 A 用這一點替候選答案排序。

對每張候選網格 $g$：
1. 數出 $n(g)$：搜尋在幾個視角中產生了 $g$；
2. 把謎題用 8 種新方式偽裝，把 $g$ 放進去當答案，在每一種偽裝下計算它的 NLL（式 2.1）；
3. 把兩者合成一個分數：
$$\mathrm{score}(g) = n(g) - \frac{1}{8}\sum_{k=1}^{8} \mathrm{NLL}_k(g)。 \tag{3.3}$$

分數最高的兩張網格就是兩次作答。

---

## 4. 通道 B：會自我檢查的程式

### 4.1 想法

直接寫網格的模型可能很接近但寫錯。*程式*不一樣：在例子的輸入上執行它，它要嘛重現例子的輸出，要嘛沒有。如果全部重現，它在測試輸入上的答案就更可能正確。

我們稱這樣的程式為*通過驗證*。

### 4.2 步驟

對每一題，通道 B 重複以下步驟。

1. **偽裝謎題。** 套用一種對稱和一種顏色命名（定義 2.4），讓模型無法認出可能看過的謎題、直接想起答案。
2. **要一段程式。** 大模型（Qwen3.8，270 億參數）收到以數字串列表示的例子與測試輸入，被要求寫一個 Python 函式 `transform(grid)`。模型會先用文字推理再寫程式，推理可能長達數萬個 token。
3. **收集候選程式。** 取最終回答裡的程式，加上推理過程中最後兩份草稿。推理被長度上限截斷時，草稿特別有用。
4. **在沙箱中執行每段程式。** 沙箱是獨立的行程，限制記憶體與時間，壞掉的程式不會影響系統其他部分。
5. **驗證。** 只有在程式**完全**重現每個例子的輸出、且對每個測試輸入都回傳合法網格時才接受。
6. **擋掉明顯的取巧。** 拒絕把例子輸出直接寫進程式碼，或把例子輸出、原封不動的測試輸入當成測試答案的程式。
7. **把測試答案的偽裝還原**，並存下來。

通道 A 最沒把握的題目先處理。沒有拿到驗證程式的題目，之後換一種偽裝再試，最多 6 次。

### 4.3 實際情況

在 30 道公開謎題上試跑 2 小時，通道 B 共開始 26 次嘗試。只有 2 次推理正常結束；11 次被長度上限（64,000 個 token）截斷，13 次被時間上限截斷。4 段程式通過驗證，其中 3 段正確。

表 4.1 從頭到尾追蹤其中四次嘗試。

| 謎題 | 經過 | 對分數的影響 |
|---|---|---|
| `dbff022c` | 推理正常結束；程式通過驗證且正確。通道 A 沒有正確答案。 | **多解 1 題** |
| `332f06d7` | 推理被截斷，但從推理中找到的一份草稿通過驗證且正確。 | **多解 1 題** |
| `2b83f449` | 被截斷；一份草稿重現了所有例子，卻給出**錯誤**的測試答案。 | 這次沒影響，但見註 5.2 |
| `53fb4810` | 通過驗證；答案已經是通道 A 兩次作答之一。 | 沒影響 |

**表 4.1** 試跑中通道 B 的四次嘗試。

和通道 A 合併後，通道 B 在三次重複試驗中都**多解了 2 題（共 30 題）**。表中也看得出主要限制：答案可靠，但推理太長，能完成的嘗試很少。

---

## 5. 合併兩個通道

> **規則 5.1（合併）。** 對每個測試輸出：
> 1. 若至少有一段通過驗證的程式，它的答案成為**第一次作答**，通道 A 的最佳答案成為**第二次作答**。（若多段程式答案不同，採用最多程式給出的答案。）
> 2. 若程式的答案已經是通道 A 兩次作答之一，就不改動。
> 3. 若沒有通過驗證的程式，保留通道 A 的兩個答案。

> **註 5.2（代價）。** 規則 5.1 取代的是通道 A 的*第二次*作答。只有在通過驗證的程式答錯、**而且**通道 A 的第二次作答本來是對的時候，才會失分。通過驗證的程式很少答錯，所以代價很小但不是零；表 4.1 的謎題 `2b83f449` 就是一段答錯的驗證程式。在我們的試跑中，沒有任何原本正確的通道 A 答案被取代。

規則 5.1 在試跑前就寫下來，所以結果無法反過來影響規則。

---

## 6. 在看不到的資料上安全執行

Kaggle 在我們看不到的謎題上執行 notebook。第 11 小時當掉就是零分。表 6.1 列出每個風險與對應的防護。

| 風險 | 防護 |
|---|---|
| 執行在結束前被中止。 | 一開始先寫出一份完整的答案檔（全部是佔位答案）。之後每次改進都一次替換整個檔案，所以隨時都有一份合法的檔案。 |
| 某一題出錯。 | 記錄錯誤後開始下一題。每張 GPU 各跑自己的行程，一個行程失敗不會拖垮其他的。 |
| 時間不夠。 | 有全域截止時間，以及隨截止時間接近而縮短的每題時間上限。 |
| 通道 B 啟動失敗。 | 通道 B 第一次嘗試之前若出任何問題，就略過通道 B，保留通道 A 的答案。 |
| 隱藏謎題和公開謎題不同。 | 每次提交前，先用題目名稱、題數、順序都不同、並模擬當機的檔案測試 notebook。 |

**表 6.1** 隱藏執行的風險與對應防護。

---

## 7. 如何量測進步

### 7.1 排行榜有雜訊

同一個方法換隨機種子提交兩次，公開分數約變動 0.9 分（一個標準差）。因此我們把小於 **2.4 分**的差距當作雜訊。用最高公開分挑最終提交，常常挑到的是運氣。

### 7.2 固定的練習題組

每個想法都在同樣 30 道公開謎題（稱為 *keys30*）上量測，並用好幾個種子重複。昂貴的步驟只跑一次並存下結果；之後較便宜的步驟在存下的結果上重播，不用 GPU。

### 7.3 修正隱藏題組較大的影響

隱藏題組的題數是 keys30 的好幾倍，但通道 B 的預算一樣是 12 小時，所以每題分到的嘗試比練習時少。我們用一個事先定好的係數 $c$ 修正：
$$\text{修正後} = A + c\,(\text{實測} - A)， \tag{7.1}$$
其中 $A$ 是只有通道 A 時的分數。對我們提交的版本，$A = 6.778$、實測 $= 8.278$、$c = 0.65$，所以

$$\text{修正後} = 6.778 + 0.65 \times (8.278 - 6.778) = 7.753 \text{ 題（共 30 題）}。$$

### 7.4 事先決定

每次較大的實驗前，我們先寫下要量什麼、什麼結果算成功、什麼時候停止。這樣看到結果後就無法重新解讀。

---

## 8. 沒有用的想法

| 想法 | 結果 | 我們的解釋 |
|---|---|---|
| 讓通道 B 推理更久（上限 96k 而非 64k token） | 修正後練習分數不變（7.84 對 7.75）；一次公開讀數低 3.19 分 | 嘗試變長，12 小時內能做的嘗試與能處理的題目變少；式 (7.1) 的係數 $c$ 從 0.65 降到 0.37。我們無法把這個原因和「錯的程式取代了正確答案」分開。 |
| 強迫通道 B 少推理 | 練習分數 7.11 對 8.78 | 難的規則需要長時間推理，砍掉後失去的比得到的多。 |
| 較低的「推理強度」設定加一輪回饋 | 約少解 1 題 | 記錄到的嘗試變多，但通過驗證的程式變少。確切原因沒有分離出來。 |
| 手寫的簡單網格操作搜尋 | 沒有變化 | 它能解的題目，很可能其他通道已經解出。 |
| 通道 A 跑兩次短的取代一次長的 | 無法判斷 | 差距在 30 題的雜訊範圍內。 |
| 不做測試時訓練的較小網格模型 | 公開 0.00 與 4.58 | 沒有測試時訓練，模型無法處理新規則。 |

**表 8.1** 測試過但放棄的想法。

---

## 9. 成績

| 提交 | 說明 | 公開分數 |
|---|---|---:|
| 57165541 | 通道 A＋通道 B——**最終選擇 1** | 32.22 |
| 56864814 | 通道 A 用不同種子跑兩次——**最終選擇 2** | 30.97 |
| 57828532 | 與最終選擇 1 同程式，換種子並固定並行數設定 | 33.89 |

**表 9.1** 主要提交。第三列與第一列的差距小於第 7.1 節的雜訊程度，所以沒有選它。

2026 年 11 月 2 日（約 18:40 UTC，3,627 隊）公開排行榜的銅牌線是 32.22。本 repo 不記錄私人排行榜分數。

---

## 附錄 A　重現提交

**在 Kaggle 上。**
1. 建立一個 notebook，使用 4 × NVIDIA L4 GPU，關閉網路。
2. 掛載競賽資料與以下輸入：
   - 模型 `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`；
   - 模型 `mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1`；
   - 資料集 `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`；
   - notebook `sorokin/pip-install-unsloth-flash-patch`；
   - 我們的 Python 3.12 下載 notebook https://www.kaggle.com/code/f0909172434/arc-2026-py312-fetch-v1（原始碼在 `runs/arc-agi-2/v13/fetch/`）。
3. 把 `runs/arc-agi-2/v13/upload-arc-2026-v13-progsynth-v2/kernel-metadata.json` 的 `id` 改成你的帳號。
4. 執行 `kaggle kernels push -p runs/arc-agi-2/v13/upload-arc-2026-v13-progsynth-v2`，再提交那個版本。

最終選擇 2 在 `runs/arc-agi-2/v8/upload-arc-2026-v8-4b-v2/`。

**在 CPU 上測試。**
```
pip install -r requirements.txt
python -m pytest runs/arc-agi-2/v13/tests
```
`tests/` 的檔案請逐一執行。除了一項需要未公開發射工具的測試（略過）與 `tests/test_arc_agi_2_fork_audit.py` 的一項已知失敗外，全部通過。

**資料。** 本 repo 不含謎題檔與模型權重。請從競賽頁或 https://github.com/arcprize/ARC-AGI-2 下載謎題，放在 `data/arc-agi-2/official-v1/`。

## 附錄 B　各部分在程式碼中的位置

| 章節 | 檔案（在 `runs/arc-agi-2/v13/src/` 下） | 主要函式 |
|---|---|---|
| 3.1 擴增 | `arc_loader.py` | `ArcDataset.augment` |
| 3.2 訓練 | `arc_solver.py` | `solve_task`、`worker` |
| 3.3 搜尋 | `arc_solver.py` | `turbo_dfs` |
| 3.4 排序 | `arc_solver.py`、`arc_decoder.py` | `calc_scores`、`score_kgmon` |
| 第 4 節 通道 B | `v13_llm.py`、`progsynth.py`、`arc_llm.py` | `run_job`、`check`、`suspicious` |
| 第 5 節 規則 5.1 | `v13_merge.py` | `fill_entry` |
| 第 6 節 防護 | `assemble.py`、`arc_v6_config.py` | `adaptive_cap` |
| 第 7 節 量測 | `../offline_v13.py`、`../c2_measure.py` | — |

**技術設定（給需要的讀者）。** LoRA rank 256，使用 rank 穩定化縮放，套用在所有注意力層與前饋層，以及嵌入層與輸出層；學習率 5 × 10⁻⁵、cosine 排程、一輪、bf16。通道 B 使用 Qwen3.8-27B-FP8，由 vLLM 0.19.0 在四張 GPU 上服務，取樣溫度 1.0、top-p 0.95、top-k 20；沙箱限制每段程式 2 GB 記憶體與 20 秒 CPU 時間。預登記與量測紀錄在 `artifacts/verification/arc-agi-2/v13/`。

## 附錄 C　專案的運作方式

本專案由 AI 程式代理（Claude Code 對話）在人類擁有者監督下完成；擁有者設定目標、核准花費與公開動作，並做最後決定。紀錄中有一些內部用語：

| 用語 | 意思 |
|---|---|
| owner session、協調者 | 負責本競賽的對話；在各競賽間分配計算時間的對話 |
| Argon、Opus、Fable、advisor | 作為獨立審查者的 AI 模型 |
| T1–T3、P0–P2 | 審查等級與審查發現的嚴重度 |
| 預登記（`prereg-*.md`） | 實驗前寫下的計畫（第 7.4 節） |
| keys30、E1–E6、C1–C3 | 練習題組（第 7.2 節）；編號的實驗 |

## 授權

Apache License 2.0，見 `LICENSE` 與 `NOTICE`。通道 A 衍生自 sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue（Apache-2.0）。所有上游程式、模型與資料列在 `THIRD_PARTY.md`。
