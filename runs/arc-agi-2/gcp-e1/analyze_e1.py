"""E1 analysis (prereg artifacts/verification/arc-agi-2/gcp-1xl4-prereg-v1.json).
Usage: .venv/bin/python runs/arc-agi-2/gcp-e1/analyze_e1.py OUT_DIR   (OUT_DIR/<F|HA|HB>-s<1..3>/{store,done,log.txt})
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/v11/src"))
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/colab-cal-v1"))
from arc_loader import ArcDataset  # noqa: E402
import assemble  # noqa: E402
import analyze  # noqa: E402

OFFSETS = {1: 0, 2: 1000, 3: 2000}
A_SEEDS = {"ARC_V8_TRAIN_AUG_SEED": 1, "ARC_V8_EVAL_AUG_SEED": 2, "ARC_V8_TRAINER_SEED": 42,
           "ARC_V8_LORA_SEED": 42, "ARC_V8_SCORE_SEED_OFFSET": 0}
B_SEEDS = {"ARC_V8_TRAIN_AUG_SEED": 17, "ARC_V8_EVAL_AUG_SEED": 29, "ARC_V8_TRAINER_SEED": 137,
           "ARC_V8_LORA_SEED": 137, "ARC_V8_SCORE_SEED_OFFSET": 7}


def seeds_ok(log, base, off):
    if not log.exists():
        return False
    starts =[json.loads(l) for l in log.read_text(errors="replace").splitlines() if '"driver_start"' in l]
    want = {k: str(v + off) for k, v in base.items()}
    return bool(starts) and all(s.get("seeds") == want for s in starts)


def main(out):
    out = Path(out)
    keys = json.loads((HERE / "build/e1-inputs-v1.manifest.json").read_text())["keys30"]
    data = ArcDataset.from_file(str(analyze.DATA / "arc-agi_evaluation_challenges.json")).change_keys(keys, keep_flags=True)
    sol = json.loads((analyze.DATA / "arc-agi_evaluation_solutions.json").read_text())
    data.replies = {k: sol[k] for k in keys}
    multi = data.split_multi_replies()
    correct = lambda d: sum(any(np.array_equal(s["solution"], multi.replies[bk][0])
                                for s in d.decoded_results.get(bk, {}).values()) for bk in multi.keys)
    per_seed, invalid = {}, []
    for s, off in OFFSETS.items():
        runs = {r: out / f"{r}-s{s}" for r in ("F", "HA", "HB")}
        for r, d in runs.items():
            n_done = len(list((d / "done").iterdir())) if (d / "done").is_dir() else 0
            if n_done < 28:
                invalid.append(f"{r}-s{s}: only {n_done}/30 tasks done")
            if not seeds_ok(d / "log.txt", B_SEEDS if r == "HB" else A_SEEDS, off):
                invalid.append(f"{r}-s{s}: seeds in log differ from prereg offsets")
        F, HA, HB = [(str(runs["F"] / "store"), "")], [(str(runs["HA"] / "store"), "")], [(str(runs["HB"] / "store"), ".b")]
        dec = {"F": assemble.load_decoder(multi, F), "HA": assemble.load_decoder(multi, HA),
               "HA+HB": assemble.load_decoder(multi, HA + HB)}
        sel = {n: d.run_selection_algo() for n, d in dec.items()}
        pooled, _ = assemble.keep_a_top1(sel["HA"], sel["HA+HB"])
        pass2 = {"F": analyze.score(analyze.select(sel["F"]), multi, keys),
                 "HA": analyze.score(analyze.select(sel["HA"]), multi, keys),
                 "H_pooled": analyze.score(pooled, multi, keys)}
        secs = {r: round(sum(v["total"] for v in analyze.timings(runs[r] / "log.txt")[0].values())) for r in runs}
        per_seed[s] = {"pass2_task_score": pass2, "D_task_score": round(pass2["H_pooled"] - pass2["F"], 3),
                       "correct_outputs_in_candidates": {n: correct(d) for n, d in dec.items()},
                       "oracle_task_score": {n: analyze.coverage(d, multi, keys) for n, d in dec.items()},
                       "seconds": secs}
    # Prereg unit: Kaggle task score (each task counts the fraction of its test outputs solved within 2 attempts).
    D =[per_seed[s]["D_task_score"] for s in OFFSETS]
    total = round(sum(D), 3)
    if invalid:
        verdict = "invalid"
    elif total >= 3 and min(D) >= 0:
        verdict = "v11_better"
    elif total <= -3:
        verdict = "v8_better"
    else:
        verdict = "inconclusive"
    summary = {"prereg": "artifacts/verification/arc-agi-2/gcp-1xl4-prereg-v1.json", "keys": len(keys),
               "outputs": len(multi.keys), "per_seed": per_seed, "D_sum": total, "invalid": invalid, "verdict": verdict}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
