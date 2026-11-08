"""Analyse the Colab one-L4 calibration run (v8 solver on a stratified public-evaluation sample).

Usage: .venv/bin/python runs/arc-agi-2/colab-cal-v1/analyze.py DOWNLOAD_DIR
DOWNLOAD_DIR holds log_a.txt, log_b.txt (optional), out_a/, out_b/ (optional) from the VM.
Prints JSON: per-task timings, a fitted time model, the projected 240-task 4xL4 rerun wall time, and
pass@2 for selection variants on the same decoded candidates (paired; the public evaluation set may be
in the model's pretraining data, so only differences between variants are meaningful).
"""
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/v8/src"))
import arc_v6_config as cfg  # noqa: E402
from arc_loader import ArcDataset  # noqa: E402
from arc_decoder import score_full_probmul_3, score_kgmon  # noqa: E402
import assemble  # noqa: E402

DATA = ROOT / "data/arc-agi-2/official-v1"
RERUN_WORKER_SECONDS = 4 * cfg.RERUN_WALL_SECONDS


def tokens(task):
    gl = lambda g: len(g) * (len(g[0]) + 1)
    return min(8192, sum(gl(p["input"]) + gl(p["output"]) + 10 for p in task["train"])
               + max(gl(q["input"]) for q in task["test"]) + 900)


def timings(log):
    text = log.read_text(errors="replace") if log.exists() else ""
    train = {k: float(s) for k, s in re.findall(r"training stats for puzzle (\S+): TrainOutput\(.*?'train_runtime': ([\d.]+)", text)}
    done = {k: float(s) for k, s in re.findall(r"\[Rank \d\] finished (\S+) in ([\d.]+)s", text)}
    failed = re.findall(r"\[Rank \d\] task (\S+) failed: (.*)", text)
    return {k: {"total": done[k], "ttt": train.get(k)} for k in done}, failed


def select(pairs):
    return {bk: v[:2] for bk, v in pairs.items()}


def score(selected, multi, keys):
    total = 0.0
    for k in keys:
        n = len(multi_keys := [b for b in multi.keys if b.rsplit("_", 1)[0] == k])
        hits = 0
        for bk in multi_keys:
            truth = multi.replies[bk][0]
            hits += any(np.array_equal(g, truth) for g in selected.get(bk, [])[:2])
        total += hits / n
    return round(total, 3)


def coverage(decoder, multi, keys):
    total = 0.0
    for k in keys:
        bks = [b for b in multi.keys if b.rsplit("_", 1)[0] == k]
        total += sum(any(np.array_equal(s["solution"], multi.replies[b][0])
                         for s in decoder.decoded_results.get(b, {}).values()) for b in bks) / len(bks)
    return round(total, 3)


def main(dl):
    dl = Path(dl)
    ev = json.loads((DATA / "arc-agi_evaluation_challenges.json").read_text())
    keys = json.loads((dl / "sample_keys.json").read_text()) if (dl / "sample_keys.json").exists() else None
    t_a, failed_a = timings(dl / "log_a.txt")
    t_b, failed_b = timings(dl / "log_b.txt")
    keys = keys or sorted(t_a)
    rows = [{"task": k, "tokens": tokens(ev[k]), "tests": len(ev[k]["test"]), **t_a[k]} for k in keys if k in t_a]
    # Fit total seconds ~ c0 + c1*L + c2*L^2 (one L4, rerun caps) and project over the evaluation distribution.
    fit = None
    if len(rows) >= 4:
        L = np.array([r["tokens"] for r in rows], float)
        y = np.array([r["total"] for r in rows], float)
        X = np.stack([np.ones_like(L), L, L ** 2], 1)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        all_L = np.array([tokens(t) for t in ev.values()], float)
        per_task = np.clip(coef[0] + coef[1] * all_L + coef[2] * all_L ** 2, 60, None)
        eval_sum = float(per_task.sum())
        fit = {"coef": [float(c) for c in coef], "eval120_worker_seconds": round(eval_sum),
               "hidden240_worker_seconds_if_like_eval": round(2 * eval_sum),
               "rerun_worker_seconds_available": RERUN_WORKER_SECONDS,
               "projected_load": round(2 * eval_sum / RERUN_WORKER_SECONDS, 3),
               "sample_sum_seconds": round(float(y.sum()))}
    data = ArcDataset.from_file(str(DATA / "arc-agi_evaluation_challenges.json"))
    data = data.change_keys(keys, keep_flags=True)
    data.replies = {k: json.loads((DATA / "arc-agi_evaluation_solutions.json").read_text())[k] for k in keys}
    multi = data.split_multi_replies()
    dec_a = assemble.load_decoder(multi, [(str(dl / "out_a"), "")])
    pooled = assemble.load_decoder(multi, [(str(dl / "out_a"), ""), (str(dl / "out_b"), ".b")])
    dec_b = assemble.load_decoder(multi, [(str(dl / "out_b"), ".b")])
    both = [k for k in keys if k in t_a and k in t_b]       # paired: tasks both passes finished
    sel_a, sel_p = dec_a.run_selection_algo(), pooled.run_selection_algo()
    v8, _ = assemble.keep_a_top1(sel_a, sel_p)
    prob_a = dec_a.run_selection_algo(score_full_probmul_3)
    alt = {}
    for bk, ranked in sel_a.items():   # attempt_2 = best probmul grid distinct from kgmon top-1
        first = ranked[0]
        other = next((g for g in prob_a.get(bk, []) if not np.array_equal(g, first)), None)
        alt[bk] = [first] + ([other] if other is not None else ranked[1:2])
    variants = {"v7_kgmon_a": select(sel_a), "v8_keep_a_top1": v8, "pooled_kgmon": select(sel_p),
                "b_only_kgmon": select(dec_b.run_selection_algo()), "a_kgmon1_probmul2": alt,
                "a_probmul": select(prob_a)}
    result = {"tasks_sampled": len(keys), "finished_a": len(t_a), "finished_b": len(t_b),
              "failed_a": failed_a, "failed_b": failed_b, "timing_rows": rows, "fit": fit,
              "paired_tasks": len(both),
              "pass2_on_paired": {n: score(v, multi, both) for n, v in variants.items()},
              "pass2_on_a_finished": {n: score(v, multi, [k for k in keys if k in t_a]) for n, v in variants.items()},
              "oracle_coverage_paired": {"a": coverage(dec_a, multi, both), "a_or_b": coverage(pooled, multi, both)}}
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main(sys.argv[1])
