"""Run the D (DSL) solver per prereg addendum (artifacts/verification/arc-agi-2/v13/prereg-arc1-c1-replay-v1.md).
  dev:    .venv/bin/python runs/arc-agi-2/v13/arc1/run_dsl.py dev  OUT.json      (training 1,000 + eval non-keys30 90)
  keys30: .venv/bin/python runs/arc-agi-2/v13/arc1/run_dsl.py keys30 OUT.jsonl   (rows {task, verified, answers, ...})
Precision rule: answers only when every train-verified program agrees on all test outputs; else unverified.
Each task runs in its own process with a hard wall limit (time_limit + 5 s); 10 s budget per task."""
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / "data/arc-agi-2/official-v1"
KEYS30 = set(json.loads((HERE.parent / "src/keys30.json").read_text()))
LIMIT = 10.0


def _work(args):
    key, task = args
    sys.path.insert(0, str(HERE))
    import dsl  # noqa: E402
    t0 = time.time()
    try:
        progs = dsl.solve([(p["input"], p["output"]) for p in task["train"]], [c["input"] for c in task["test"]], LIMIT)
    except Exception as e:  # noqa: BLE001
        progs, err = [], f"{type(e).__name__}: {e}"[:200]
    else:
        err = ""
    dt = time.time() - t0
    outs = {json.dumps(o) for _, o in progs}
    verified = len(outs) == 1 and dt <= LIMIT + 1.0  # solver enforces its own deadline; hard kill at LIMIT+5
    return {"task": key, "verified": verified, "answers": json.loads(next(iter(outs))) if verified else None,
            "programs": [n for n, _ in progs][:5], "n_programs": len(progs), "n_distinct": len(outs),
            "seconds": round(dt, 2), "error": err}


PROCS = 4  # light-check default outside the local-compute night window


def run(tasks, procs=None):
    procs = procs or PROCS
    out = []
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=20) as pool:
        pending = {k: pool.apply_async(_work, ((k, t),)) for k, t in tasks.items()}
        for k, r in pending.items():
            try:
                out.append(r.get(timeout=LIMIT + 5))
            except Exception as e:  # noqa: BLE001  (hard timeout / crash)
                out.append({"task": k, "verified": False, "answers": None, "programs": [], "n_programs": 0,
                            "n_distinct": 0, "seconds": LIMIT + 5, "error": f"hard {type(e).__name__}"})
    return out


def dev(dest, splits="evaluation,training"):
    """splits: comma list of evaluation (non-keys30) / training / training:N (first N sorted training tasks)."""
    res = {"limit_s": LIMIT, "dsl_sha256": hashlib.sha256((HERE / "dsl.py").read_bytes()).hexdigest(), "splits": {}}
    for spec in splits.split(","):
        split, _, n = spec.partition(":")
        skip = KEYS30 if split == "evaluation" else set()
        chal = json.loads((DATA / f"arc-agi_{split}_challenges.json").read_text())
        sol = json.loads((DATA / f"arc-agi_{split}_solutions.json").read_text())
        tasks = {k: v for k, v in sorted(chal.items()) if k not in skip}
        if n:
            tasks = dict(list(tasks.items())[:int(n)])
        rows = run(tasks)
        ver = [r for r in rows if r["verified"]]
        right = [r for r in ver if r["answers"] == sol[r["task"]]]
        multi_disagree = sum(1 for r in rows if r["n_distinct"] > 1)
        secs = sorted(r["seconds"] for r in rows)
        res["splits"][spec] = {"tasks": len(rows), "verified": len(ver), "solved": len(right),
                                "precision": round(len(right) / len(ver), 3) if ver else None,
                                "withheld_disagree": multi_disagree,
                                "found_unverified": sum(1 for r in rows if r["n_programs"] > 0 and not r["verified"]), "p95_s": secs[int(0.95 * (len(secs) - 1))],
                                "total_s": round(sum(secs), 1), "errors": sum(bool(r["error"]) for r in rows),
                                "solved_tasks": sorted(r["task"] for r in right),
                                "wrong": [{"task": r["task"], "programs": r["programs"]} for r in ver if r not in right]}
        print(spec, {k: v for k, v in res["splits"][spec].items() if k not in ("solved_tasks", "wrong")}, flush=True)
    Path(dest).write_text(json.dumps(res, indent=1) + "\n")


def keys30(dest):
    chal = json.loads((DATA / "arc-agi_evaluation_challenges.json").read_text())
    rows = run({k: chal[k] for k in sorted(KEYS30)})
    Path(dest).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("keys30 verified", sum(r["verified"] for r in rows), "of", len(rows),
          "dsl_sha256", hashlib.sha256((HERE / "dsl.py").read_bytes()).hexdigest()[:8])


if __name__ == "__main__":
    {"dev": dev, "keys30": keys30}[sys.argv[1]](*sys.argv[2:])
