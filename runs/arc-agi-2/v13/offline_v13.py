"""v13 offline counterfactual on keys30 (zero GPU): per E1 seed, HA (v11 half-cost pass A) top-2 vs the v13 fill rule
with P0b program answers: a verified program's test answer -> attempt_1, HA top-1 -> attempt_2 (HA top-2 when the
program answer equals HA top-1). Also HA kgmon confidence per task (v9 task_confidence) to check whether the
program-only wins were tasks HA is unsure of.
  .venv/bin/python runs/arc-agi-2/v13/offline_v13.py runs/arc-agi-2/gcp-e1/out/e1 artifacts/verification/arc-agi-2/v13-offline-v1.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/gcp-e1"))
import analyze_e1 as E  # noqa: E402  (puts v11/src and colab-cal-v1 on sys.path)
from analyze_e1 import ArcDataset, analyze, assemble  # noqa: E402
import arc_v6_config as cfg  # noqa: E402

out, dest = Path(sys.argv[1]), Path(sys.argv[2])
keys = json.loads((E.HERE / "build/e1-inputs-v1.manifest.json").read_text())["keys30"]
chal = json.loads((analyze.DATA / "arc-agi_evaluation_challenges.json").read_text())
sol = json.loads((analyze.DATA / "arc-agi_evaluation_solutions.json").read_text())
data = ArcDataset.from_file(str(analyze.DATA / "arc-agi_evaluation_challenges.json")).change_keys(keys, keep_flags=True)
data.replies = {k: sol[k] for k in keys}
multi = data.split_multi_replies()


def programs(tag):
    """task -> list of verified program rows (answers inverted to the original frame), job order."""
    res = {}
    for line in open(ROOT / f"runs/arc-agi-2/p0b-colab/out/out/{tag}-calls.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "program" and r.get("verified"):
            res.setdefault(r["task"], []).append(r)
    return res


def eq(a, b):
    return a is not None and b is not None and np.array_equal(np.array(a), np.array(b))


def score(attempts):
    """attempts: bk -> list of grids (first two count). Returns per-task score dict."""
    r = {}
    for k in keys:
        n = len(sol[k])
        r[k] = sum(any(eq(g, sol[k][i]) for g in attempts.get(f"{k}_{i}", [])[:2]) for i in range(n)) / n
    return r


def v13_fill(ha_sel, progs):
    att = {}
    for k in keys:
        for i in range(len(sol[k])):
            bk = f"{k}_{i}"
            ha = [np.array(g).tolist() for g in ha_sel.get(bk, [])[:2]]
            rows = progs.get(k, [])
            if rows:
                p = rows[0]["answers"][i]
                rest = [g for g in ha if not eq(g, p)]
                att[bk] = [p] + rest[:1]
            else:
                att[bk] = ha
    return att


res = {"keys": len(keys), "seeds": {}}
prog = {t: programs(t) for t in ("q38", "q36a3b")}
for s in (1, 2, 3):
    store = out / f"HA-s{s}" / "store"
    dA = assemble.load_decoder(multi, [(str(store), "")])
    ha_sel = analyze.select(dA.run_selection_algo())
    ha = score(ha_sel)
    v13 = score(v13_fill(ha_sel, prog["q38"]))
    both = {k: prog["q38"].get(k, []) + prog["q36a3b"].get(k, []) for k in keys}
    v13b = score(v13_fill(ha_sel, {k: v for k, v in both.items() if v}))
    conf = cfg.task_confidence(keys, str(store), chal, 1e18)
    res["seeds"][s] = {"HA": round(sum(ha.values()), 3), "v13_q38": round(sum(v13.values()), 3),
                       "v13_q38_then_q36": round(sum(v13b.values()), 3),
                       "changed_tasks": {k: [ha[k], v13[k]] for k in keys if ha[k] != v13[k]},
                       "conf": {k: (None if conf[k] == float("-inf") else round(conf[k], 3)) for k in keys}}
for arm in ("HA", "v13_q38", "v13_q38_then_q36"):
    v = [res["seeds"][s][arm] for s in (1, 2, 3)]
    res[arm] = {"per_seed": v, "mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v, ddof=1)), 3)}
res["F_reference"] = {"mean": 7.61, "per_seed": [9.0, 5.83, 8.0], "source": "gcp-e1-result-v1.json"}
res["program_tasks_q38"] = {k: [{"rep": r["rep"], "correct": r["correct"]} for r in v] for k, v in prog["q38"].items()}
res["confident_threshold"] = cfg.PASS_B_CONFIDENT
res["caveats"] = ["P0b q38 completed 26/60 round-1 jobs in 40 min (1 Colab GPU); programs exist only for tasks whose job "
                  "finished, so this is a lower bound on coverage and an upper bound on nothing else",
                  "keys30 are public-eval tasks; P0b used colour/rotation augmentation but contamination of Qwen3.8 on "
                  "public ARC-AGI-2 eval cannot be excluded; hidden test only verifiable on the public LB",
                  "single program sample per (task, rep); program precision q38 4/5 on verified"]
dest.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({a: res[a] for a in ("HA", "v13_q38", "v13_q38_then_q36", "F_reference")}, indent=1))
for s in (1, 2, 3):
    print(s, "changed", res["seeds"][s]["changed_tasks"])
c = res["seeds"][1]["conf"]
print("conf s1 (program tasks):", {k: [res["seeds"][s]["conf"][k] for s in (1, 2, 3)] for k in prog["q38"]})
print("conf s1 all sorted:", sorted(((v if v is not None else -1e9), k) for k, v in c.items()))
