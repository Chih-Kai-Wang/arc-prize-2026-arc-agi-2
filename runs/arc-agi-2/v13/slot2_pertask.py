"""Slot-2 prep (zero GPU): per-task keys30 matrices for v8 (E1 F, 3 seeds), v11 (E1 H pooled) and v13 v2 (E1 HA seed s +
E3 program rep r, 2x3 cells, via the replay files' per_task_delta) -> task-level and same-seed correlations used by the
slot-2 E[max] draft. E1 F-s and HA-s share seed_offset s (driver_e1.py), like v8 and v13's HA share fixed pass-A seeds.
  .venv/bin/python runs/arc-agi-2/v13/slot2_pertask.py artifacts/verification/arc-agi-2/v13/slot2-pertask-v1.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/gcp-e1"))
import analyze_e1 as E  # noqa: E402
from analyze_e1 import ArcDataset, analyze, assemble  # noqa: E402

V = ROOT / "artifacts/verification/arc-agi-2"
out = ROOT / "runs/arc-agi-2/gcp-e1/out/e1"
keys = json.loads((E.HERE / "build/e1-inputs-v1.manifest.json").read_text())["keys30"]
sol = json.loads((analyze.DATA / "arc-agi_evaluation_solutions.json").read_text())
data = ArcDataset.from_file(str(analyze.DATA / "arc-agi_evaluation_challenges.json")).change_keys(keys, keep_flags=True)
data.replies = {k: sol[k] for k in keys}
multi = data.split_multi_replies()
pert = json.loads((V / "gcp-e1-pertask-v1.json").read_text())


def per_task(selected):  # same as e1_pertask.per_task
    r = {}
    for k in keys:
        bks = [b for b in multi.keys if b.rsplit("_", 1)[0] == k]
        r[k] = sum(any(np.array_equal(g, multi.replies[b][0]) for g in selected.get(b, [])[:2]) for b in bks) / len(bks)
    return r


HA = {s: per_task(analyze.select(assemble.load_decoder(multi, [(str(out / f"HA-s{s}" / "store"), "")]).run_selection_algo()))
      for s in (1, 2, 3)}
reps = {r: json.loads((V / f"v13/replay-e3-rep{r}.json").read_text()) for r in (1, 2)}
for r, rp in reps.items():  # HA must reproduce the replay's own "before"
    for s in (1, 2, 3):
        assert abs(sum(HA[s].values()) - rp["seeds"][str(s)]["before"]) < 1e-3, (r, s)
V13 = {(r, s): {k: HA[s][k] + reps[r]["seeds"][str(s)]["per_task_delta"].get(k, 0.0) for k in keys}
       for r in (1, 2) for s in (1, 2, 3)}
F = {s: {k: pert[k]["F"][s - 1] for k in keys} for s in (1, 2, 3)}
H = {s: {k: pert[k]["H"][s - 1] for k in keys} for s in (1, 2, 3)}


def vec(d):
    return np.array([d[k] for k in keys])


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None


mF = np.mean([vec(F[s]) for s in (1, 2, 3)], 0)
mH = np.mean([vec(H[s]) for s in (1, 2, 3)], 0)
mA = np.mean([vec(HA[s]) for s in (1, 2, 3)], 0)
m13 = np.mean([vec(V13[c]) for c in V13], 0)
# same-seed vs cross-seed per-task agreement: does a shared seed make F and v13 succeed on the same tasks beyond the
# task-level rates? (residuals after removing each method's per-task mean)
same, cross = [], []
for (r, s), v in V13.items():
    for s2 in (1, 2, 3):
        c = corr(vec(F[s2]) - mF, vec(v) - m13)
        (same if s2 == s else cross).append(c)
res = {
    "keys": len(keys),
    "totals": {"F_v8": [round(sum(F[s].values()), 3) for s in (1, 2, 3)],
               "H_v11": [round(sum(H[s].values()), 3) for s in (1, 2, 3)],
               "HA": [round(sum(HA[s].values()), 3) for s in (1, 2, 3)],
               "v13_cells": {f"rep{r}-s{s}": round(sum(v.values()), 3) for (r, s), v in V13.items()}},
    "task_level_corr": {"v13_v8": corr(m13, mF), "v13_v11": corr(m13, mH), "v8_v11": corr(mF, mH),
                        "v13_HA": corr(m13, mA), "HA_v8": corr(mA, mF)},
    "residual_corr_F_vs_v13": {"same_seed_mean": float(np.mean([c for c in same if c is not None])),
                               "cross_seed_mean": float(np.mean([c for c in cross if c is not None])),
                               "n_same": len(same), "n_cross": len(cross)},
    "per_task": {k: {"v8": round(float(mF[i]), 3), "v11": round(float(mH[i]), 3), "HA": round(float(mA[i]), 3),
                     "v13": round(float(m13[i]), 3)} for i, k in enumerate(keys) if mF[i] or mH[i] or m13[i]},
    "v13_only_tasks": [k for i, k in enumerate(keys) if m13[i] > 0 and mF[i] == 0],
    "v8_only_tasks": [k for i, k in enumerate(keys) if mF[i] > 0 and m13[i] == 0],
    "solved_any": {"v8": int((mF > 0).sum()), "v13": int((m13 > 0).sum()), "both": int(((mF > 0) & (m13 > 0)).sum())},
}
Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=1))
for k, v in res["per_task"].items():
    print(k, v)
