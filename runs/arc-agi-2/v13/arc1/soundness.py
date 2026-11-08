"""Prereg section 4: per-invariant false-reject rate of the TRUE test output, on the training set (1,000 tasks) and the
evaluation tasks outside keys30 (90). keys30 tasks are skipped before any field is read. Invariants above 1% are dropped
(written to the output's "active" list, which the keys30 replay must use unchanged).
  .venv/bin/python runs/arc-agi-2/v13/arc1/soundness.py artifacts/verification/arc-agi-2/v13/arc1-soundness-v1.json"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import invariants as I  # noqa: E402

DATA = ROOT / "data/arc-agi-2/official-v1"
KEYS30 = set(json.loads((HERE.parent / "src/keys30.json").read_text()))
MAX_RATE = 0.01


def tally(split, skip=()):
    chal = json.loads((DATA / f"arc-agi_{split}_challenges.json").read_text())
    sol = json.loads((DATA / f"arc-agi_{split}_solutions.json").read_text())
    app, bad, tasks_bad = defaultdict(int), defaultdict(int), defaultdict(list)
    n = 0
    for k in sorted(chal):
        if k in skip:
            continue
        n += 1
        t = chal[k]
        inv = I.infer([(p["input"], p["output"]) for p in t["train"]])
        for idx, case in enumerate(t["test"]):
            for name in inv:
                app[name] += 1
            for name in I.violations(inv, case["input"], sol[k][idx]):
                bad[name] += 1
                tasks_bad[name].append(f"{k}_{idx}")
    return n, app, bad, tasks_bad


def main(dest):
    res = {"max_rate": MAX_RATE, "splits": {}}
    tot_app, tot_bad = defaultdict(int), defaultdict(int)
    for split, skip in (("training", ()), ("evaluation", KEYS30)):
        n, app, bad, tb = tally(split, skip)
        res["splits"][split] = {"tasks": n, "per_invariant": {k: {"applicable": app[k], "violations": bad[k],
                                                                  "examples": tb[k][:5]} for k in sorted(app)}}
        for k in app:
            tot_app[k] += app[k]
            tot_bad[k] += bad[k]
    res["combined"] = {k: {"applicable": tot_app[k], "violations": tot_bad[k],
                           "rate": round(tot_bad[k] / tot_app[k], 4)} for k in sorted(tot_app)}
    res["active"] = sorted(k for k, v in res["combined"].items() if v["rate"] <= MAX_RATE)
    res["dropped"] = sorted(k for k, v in res["combined"].items() if v["rate"] > MAX_RATE)
    res["invariants_sha256"] = __import__("hashlib").sha256((HERE / "invariants.py").read_bytes()).hexdigest()
    Path(dest).write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["combined"].items():
        print(f"{k:24s} applicable {v['applicable']:5d}  violations {v['violations']:4d}  rate {v['rate']:.4f}")
    print("active", res["active"])
    print("dropped", res["dropped"])


if __name__ == "__main__":
    main(sys.argv[1])
