"""Descriptive (not pre-registered) re-estimate of the E3 coverage adjustment with the rerun's real LLM job order.

prereg-e3-seeds-v1.md adjusts E3 as HA + (raw - HA) * c, i.e. as if the program channel reached a random 65% of tasks;
the rerun actually orders jobs by NVARC pass-A confidence ascending (v13_llm.job_order, sure tasks last). Here: per E1 HA
seed, order keys30 the same way (arc_v6_config.task_confidence on that seed's HA store; key order breaks ties), count only
the per-task deltas of replay-e3-rep{1,2}.json that fall in the first round(c*30) tasks, and report 6-cell means for
c = 0.5 / 0.65 / 0.8. keys30 is not a random sample of the hidden set, so this is a sensitivity check, not an estimate
that replaces the pre-registered 7.753.
  .venv/bin/python runs/arc-agi-2/v13/experiments/coverage_order_e3.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/v13/src3"))
import arc_v6_config as cfg  # noqa: E402

V13 = ROOT / "artifacts/verification/arc-agi-2/v13"
E1 = ROOT / "runs/arc-agi-2/gcp-e1/out/e1"
ARM = os.environ.get("ARM", "HA")
CONFIDENT = 5.0  # v13_llm.CONFIDENT
rep_files = {r: json.loads((V13 / "poe-check" / f"replay-{ARM}-e3-rep{r}.json").read_text()) for r in (1, 2)}
keys = json.loads((ROOT / "runs/arc-agi-2/v13/src3/keys30.json").read_text())
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/gcp-e1"))
from analyze_e1 import analyze  # noqa: E402
chal = json.loads((analyze.DATA / "arc-agi_evaluation_challenges.json").read_text())

out = {"keys": len(keys), "cells": {}, "means": {}}
CS = (0.25, 0.3, 0.4, 0.45, 0.5, 0.65, 0.8, 1.0)
cells = {c: [] for c in CS}
for s in (1, 2, 3):
    conf = cfg.task_confidence(keys, str(E1 / f"{ARM}-s{s}" / "store"), chal, time.time() + 600)
    rank = {k: i for i, k in enumerate(keys)}
    unsure = [k for k in keys if conf.get(k, float("-inf")) < CONFIDENT]
    sure = [k for k in keys if k not in unsure]
    order = sorted(unsure, key=lambda k: (conf.get(k, float("-inf")), rank[k])) + sure
    for r in (1, 2):
        seed = rep_files[r]["seeds"][str(s)]
        before, delta = seed["before"], seed["per_task_delta"]
        for c in cells:
            covered = set(order[:round(c * len(keys))])
            cells[c].append(before + sum(v for k, v in delta.items() if k in covered))
        out["cells"][f"s{s}-rep{r}"] = {"before": before, "gain_tasks": {k: round(v, 3) for k, v in delta.items()},
                                         "gain_positions": {k: order.index(k) for k in delta},
                                         "sure_tasks": len(sure)}
ha = float(np.mean([rep_files[1]["seeds"][str(s)]["before"] for s in (1, 2, 3)]))
for c, v in cells.items():
    out["means"][str(c)] = {"ordered": round(float(np.mean(v)), 3), "random_formula": None}
raw = out["means"]["1.0"]["ordered"]
for c in CS[:-1]:
    out["means"][str(c)]["random_formula"] = round(ha + (raw - ha) * c, 3)
out["ha_mean"] = round(ha, 3)
dest = V13 / "poe-check" / f"coverage-order-{ARM}-e3.json"
dest.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["means"], indent=1), "HA", out["ha_mean"])
for name, cell in out["cells"].items():
    print(name, cell["gain_positions"], {k: v for k, v in cell["gain_tasks"].items()})
