"""F2 of the v2 pre-submit API review (api-reviews/20261014T022729Z-*): replay the shipped merge (v13_merge.merge, the
exact function the notebook runs) onto a REAL NVARC base — E1's HA (v11 half-cost pass A) top-2 for keys30, per seed —
with the v2 dev run's verified programs (v13-programs.jsonl, answers already in the original frame). Reports keys30
task-score before/after, how each touched output changed (already in NVARC top-2 / filled a placeholder / displaced
attempt_2), and whether any displacement lost a correct NVARC answer. Ship if the score change is non-negative.
  .venv/bin/python runs/arc-agi-2/v13/replay_v2_merge.py runs/arc-agi-2/gcp-e1/out/e1 \
      runs/arc-agi-2/v6/output-arc-2026-v13-progsynth-v2/v13-programs.jsonl artifacts/verification/arc-agi-2/v13/replay-v2-merge.json
E3 (one program seed at a time; the E3 jsonl holds reps 0/1/2 mixed): append `--rep N` to replay only rows with rep N.
v3 paired delta (prereg amendment): append `--exclude-source forced` to replay without the forced-source rows."""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/gcp-e1"))
import analyze_e1 as E  # noqa: E402  (puts v11/src and colab-cal-v1 on sys.path)
from analyze_e1 import ArcDataset, analyze, assemble  # noqa: E402
sys.path.insert(0, str(HERE / "src"))
import v13_merge as M  # noqa: E402

ARM = os.environ.get("ARM", "HA")  # E1 arm: HA (v11 half cost) or F (v8 full cost)
out, programs, dest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
opts = dict(zip(sys.argv[4::2], sys.argv[5::2]))
if len(sys.argv) % 2 or set(opts) - {"--rep", "--exclude-source"}:
    raise SystemExit(__doc__)
rep = int(opts["--rep"]) if "--rep" in opts else None
exclude = opts.get("--exclude-source")
keys = json.loads((E.HERE / "build/e1-inputs-v1.manifest.json").read_text())["keys30"]
chal = json.loads((analyze.DATA / "arc-agi_evaluation_challenges.json").read_text())
sol = json.loads((analyze.DATA / "arc-agi_evaluation_solutions.json").read_text())
data = ArcDataset.from_file(str(analyze.DATA / "arc-agi_evaluation_challenges.json")).change_keys(keys, keep_flags=True)
data.replies = {k: sol[k] for k in keys}
multi = data.split_multi_replies()
test = {k: chal[k] for k in keys}
if rep is not None or exclude:  # same loader as the notebook, on the selected rows only
    import tempfile
    rows = [line for line in programs.read_text().splitlines() if line.strip()
            and (rep is None or json.loads(line).get("rep") == rep)
            and (exclude is None or json.loads(line).get("source") != exclude)]
    tmp = Path(tempfile.mkdtemp()) / f"rep{rep}.jsonl"
    tmp.write_text("".join(line + "\n" for line in rows))
    programs = tmp
results = {k: v for k, v in M.load_results(programs).items() if k in test}


def task_score(sub):
    return {k: sum(any(g == sol[k][i] for g in (e["attempt_1"], e["attempt_2"])) for i, e in enumerate(sub[k])) / len(sol[k])
            for k in keys}


res = {"keys": len(keys), "rep": rep, "exclude_source": exclude, "program_tasks": sorted(results), "seeds": {}}
for s in (1, 2, 3):
    dA = assemble.load_decoder(multi, [(str(out / f"{ARM}-s{s}" / "store"), "")])
    sel = analyze.select(dA.run_selection_algo())
    base = {}
    for k in keys:
        entries = []
        for i in range(len(test[k]["test"])):
            g = [np.array(x).tolist() for x in sel.get(f"{k}_{i}", [])[:2]]
            g += [M.PLACEHOLDER] * (2 - len(g))
            entries.append({"attempt_1": g[0], "attempt_2": g[1]})
        base[k] = entries
    M.validate(test, base)
    merged, changed = M.merge(test, base, results)
    M.validate(test, merged)
    before, after = task_score(base), task_score(merged)
    kinds = []
    for k in results:
        for i, (b, m) in enumerate(zip(base[k], merged[k])):
            real = [g for g in (b["attempt_1"], b["attempt_2"]) if g != M.PLACEHOLDER]
            kind = ("unchanged_in_top2" if m == b else "filled_placeholder" if len(real) < 2 else "displaced_attempt2")
            lost = [a for a in ("attempt_1", "attempt_2") if b[a] == sol[k][i] and sol[k][i] not in (m["attempt_1"], m["attempt_2"])]
            kinds.append({"task": k, "i": i, "kind": kind, "program_correct": m["attempt_1"] == sol[k][i],
                          "nvarc_had_correct": sol[k][i] in real, "lost_correct_nvarc": bool(lost)})
    res["seeds"][s] = {"before": round(sum(before.values()), 3), "after": round(sum(after.values()), 3),
                       "delta": round(sum(after.values()) - sum(before.values()), 3), "changed_outputs": changed,
                       "per_task_delta": {k: after[k] - before[k] for k in keys if after[k] != before[k]}, "outputs": kinds}
d = [res["seeds"][s]["delta"] for s in (1, 2, 3)]
res["delta_mean"] = round(float(np.mean(d)), 3)
res["lost_correct_nvarc_any"] = any(o["lost_correct_nvarc"] for s in (1, 2, 3) for o in res["seeds"][s]["outputs"])
res["ship_if"] = "delta non-negative on every seed and no lost correct NVARC answer (API review F2)"
res["pass"] = all(x >= 0 for x in d) and not res["lost_correct_nvarc_any"]
dest.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({s: {k: v for k, v in res["seeds"][s].items() if k != "outputs"} for s in (1, 2, 3)}, indent=1))
for s in (1, 2, 3):
    print(s, [(o["task"], o["kind"], o["program_correct"], o["nvarc_had_correct"], o["lost_correct_nvarc"])
              for o in res["seeds"][s]["outputs"]])
print("pass", res["pass"], "delta_mean", res["delta_mean"])
