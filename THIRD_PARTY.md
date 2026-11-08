# Third-party code, models, data and ideas

"License" below is only what we could verify; "not verified" means we did not find or could not read a license statement. Nothing in the *Not redistributed* rows is contained in this repository; those items are attached to the Kaggle notebooks at run time.

## Code contained in this repository

| Upstream | URL | What we used / where | License |
|---|---|---|---|
| sorokin (ivan) & CPMP — *arc2-qwen3-unsloth-flash-lora-batch4-queue*, notebook version 10 | https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue | The whole NVARC-style Qwen3-4B pipeline (per-task rank-256 rsLoRA test-time training on augmented examples, 16-view turbo-DFS decoding, augmentation re-scoring, candidate selection). Verbatim cells in `runs/arc-agi-2/v6/author-v10/`; modified derivatives in `runs/arc-agi-2/{v6,v7,v8,v9,v11}/src/` and `runs/arc-agi-2/v13/src*/` (see NOTICE). Both final submissions are built on it. | Apache-2.0 (stated on the Kaggle page) |
| mikelou1 — *arc-agi2-lb33-89-minimal-perfpatch* | https://www.kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch | Forked as-is (byte-identical notebook, SHA-256 `ee49384a…`) in `runs/arc-agi-2/fork-perfpatch/` (only `kernel-metadata.json` is ours); public score of our fork 30.14. A sorokin derivative with a GPU-side logits performance patch. | Apache-2.0 (stated on the Kaggle page, checked 2026-11-08) |
| evgendvorkin — *arc-2-2026-4-4-local-boyko-nvarc* (V22, SHA-256 `3869a268…`) | https://www.kaggle.com/code/evgendvorkin/arc-2-2026-4-4-local-boyko-nvarc | Forked with two edits (commit-mode time budget; expert promotion default off) by `runs/arc-agi-2/fork-boyko/build_fork.py`; notebook and extracted cells in `runs/arc-agi-2/fork-boyko/`. Public score of our fork 30.56 (not selected). Its own credits: Konstantin Boyko — https://www.kaggle.com/code/konstantinboyko/failed-in-aimo-on-docker-v169-v03 ; Koushik Rudra — https://www.kaggle.com/code/koushikrudra/failed-in-aimo ; Mayank Raj — https://www.kaggle.com/code/mayankraaz/arc-agi-2-triple-pass-nvarc-sota-35-9-lb | Apache-2.0 (stated on the Kaggle page, checked 2026-11-08) |

`build_fork.py` expects the upstream evgendvorkin notebook at `runs/arc-agi-2/v6/external-src/evgendvorkin__arc-2-2026-4-4-local-boyko-nvarc/` (not included; pull it from Kaggle).

## Ideas and small snippets credited

| Upstream | URL | What |
|---|---|---|
| NVARC team (Ivan Sorokin, Jean-François Puget) — ARC Prize 2025 1st place | https://arcprize.org/blog/arc-prize-2025-results-analysis | The NVARC method the sorokin notebook implements. Method credit only; we copied no code from the NVARC repository. |
| The ARChitects — ARC Prize 2024 | https://github.com/da-fr/arc-prize-2024 | Origin of the DFS sampling / augmentation-scoring line of work that the NVARC-style pipeline builds on. Apache-2.0. No code copied directly from that repository. |
| koushikrudra — *Failed in AIMO* | https://www.kaggle.com/code/koushikrudra/failed-in-aimo | Widely forked NVARC derivative studied during triage; credited by the Boyko fork. |
| luxluxshan — *arc2-nvarc-v1* | https://www.kaggle.com/code/luxluxshan/arc2-nvarc-v1 | "Half-cost" TTT width (8 train augmentations × 8 geometries = 64 sequences) adopted in v11 (`runs/arc-agi-2/v11/src/arc_v6_config.py:28`) and therefore in v13's NVARC pass. |
| delongyang — *arc-agi-2-inference-qwen3-8-27b-fp8-within-4-hours* | https://www.kaggle.com/code/delongyang/arc-agi-2-inference-qwen3-8-27b-fp8-within-4-hours | vLLM-on-L4 setup details for Qwen3.8-27B-FP8: the libcuda pin and the MTP-drafter attention-backend fix (`runs/arc-agi-2/p0-qwen/src/p0_probe.py:172-206`, `runs/arc-agi-2/v13/src*/v13_llm.py`). Our code comments record it as an Apache-2.0 notebook (not re-verified). |

## Not redistributed: models, datasets, kernels used at run time

| Item | URL | Used by | License |
|---|---|---|---|
| sorokin/qwen3_4b_grids15_sft139 (Transformers, bfloat16, v1) | https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139 | NVARC pass in v6–v13 | Apache 2.0 (Kaggle model instance licenseName, read 2026-10-12, recorded in `artifacts/verification/arc-agi-2/v6/audit-arc-2026-fork-perfpatch-v1.json:58`) |
| sorokin/pip-install-unsloth-flash-patch (kernel) | https://www.kaggle.com/code/sorokin/pip-install-unsloth-flash-patch | Unsloth install for the NVARC pass | Notebook: Apache-2.0 (stated on the Kaggle page, checked 2026-11-08); Unsloth itself: see https://github.com/unslothai/unsloth |
| mikedan7/qwen3-8-27b-fp8-official (PyTorch, hf-fp8, v1) — mirror of Qwen/Qwen3.8-27B-FP8, HF revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` (file hashes in `runs/arc-agi-2/v13/src/qwen38_fp8_sha256.json`) | https://www.kaggle.com/models/mikedan7/qwen3-8-27b-fp8-official | v13 program-synthesis LLM | Qwen3.8-27B and its FP8 release recorded as Apache-2.0 from Hugging Face (`artifacts/verification/arc-agi-2/new-approach-proposal-v1.md:223`); the Kaggle mirror itself not verified |
| mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3 (dataset) | https://www.kaggle.com/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3 | vLLM 0.19.0 wheels for the v13 LLM phase | not verified (vLLM: https://github.com/vllm-project/vllm) |
| python-build-standalone 3.12.14 (20260924) | https://github.com/astral-sh/python-build-standalone | Python 3.12 runtime for the vLLM venv, downloaded by our kernel https://www.kaggle.com/code/f0909172434/arc-2026-py312-fetch-v1 (source: `runs/arc-agi-2/v13/fetch/`; SHA-256 checked) | not verified |
| konstantinboyko/qwen3-4b-bfloat16-v02-01-01, konstantinboyko/unsloth-2025-9-7-torch-2-8-0-py3-12-patched, konstantinboyko/flash-attn-2-8-2 | https://www.kaggle.com/konstantinboyko | Boyko fork only | not verified |
| sorokin/qwen3_2b_grids15_sft141; qwen-lm/qwen2.5-coder 7b-instruct | Kaggle Models | early pipelines (not included here; mentioned in results only) | not verified |
| ARC-AGI-2 tasks | https://github.com/arcprize/ARC-AGI-2 and the Kaggle competition data page | evaluation and tests (place under `data/arc-agi-2/official-v1/`) | Apache-2.0 ("Copyright 2019 ARC Prize Foundation"); **no task files are included in this repository** |

## Generated / excluded
Kaggle kernel outputs (logs, `unsloth_compiled_cache/`, submissions), copies of ~25 other public notebooks we pulled for triage, and the competition data were left out of this repository.
