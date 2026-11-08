"""ARC-1/C1 keys30 replay (prereg artifacts/verification/arc-agi-2/v13/prereg-arc1-c1-replay-v1.md section 2-3).
Cells = E3 rep {1,2} x E1 HA seed {1,2,3}. R = stable invariant filter of the full kgmon list (c=1); F = invariant filter
of LLM program answers (c=0.65); D = DSL answers (c=1; --dsl JSONL of {task, answers}, verified rows only).
adj = mean6[S_D + 0.65 * (S_all - S_D)]. --off R,F,D switches components off (all off must give 6.778/8.278/7.753).
  .venv/bin/python runs/arc-agi-2/v13/arc1/replay_arc1.py artifacts/verification/arc-agi-2/v13/arc1-soundness-v1.json \
      artifacts/verification/arc-agi-2/v13/arc1-replay-v1.json [--off R,F,D] [--dsl FILE]"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V13 = HERE.parent
ROOT = V13.parents[2]
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/gcp-e1"))
import analyze_e1 as E  # noqa: E402
from analyze_e1 import ArcDataset, analyze, assemble  # noqa: E402
sys.path.insert(0, str(V13 / "src"))
import v13_merge as M  # noqa: E402
sys.path.insert(0, str(HERE))
import invariants as I  # noqa: E402

C = 0.65
C96 = 0.37
OUT_E1 = ROOT / "runs/arc-agi-2/gcp-e1/out/e1"
RUNS = ROOT / "runs/arc-agi-2/v6"
LLM_LOGS = {"E3": "output-arc-2026-v13-e3-seeds-v1", "E6": "output-arc-2026-v13-e6-v4seeds-v1",
            "v2": "output-arc-2026-v13-progsynth-v2", "v3": "output-arc-2026-v13-progsynth-v3",
            "v4": "output-arc-2026-v13-progsynth-v4-c96"}


def rows(name):
    p = RUNS / LLM_LOGS[name] / "v13-programs.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main(sound, dest, off=(), dsl=None):
    active = set(json.loads(Path(sound).read_text())["active"])
    keys = json.loads((E.HERE / "build/e1-inputs-v1.manifest.json").read_text())["keys30"]
    chal = json.loads((analyze.DATA / "arc-agi_evaluation_challenges.json").read_text())
    sol = json.loads((analyze.DATA / "arc-agi_evaluation_solutions.json").read_text())
    data = ArcDataset.from_file(str(analyze.DATA / "arc-agi_evaluation_challenges.json")).change_keys(keys, keep_flags=True)
    data.replies = {k: sol[k] for k in keys}
    multi = data.split_multi_replies()
    test = {k: chal[k] for k in keys}
    inv = {k: I.infer([(p["input"], p["output"]) for p in chal[k]["train"]]) for k in keys}

    def viol(k, i, g):
        return I.violations(inv[k], chal[k]["test"][i]["input"], g, active)

    def task_score(sub):
        return sum(sum(any(g == sol[k][i] for g in (e["attempt_1"], e["attempt_2"])) for i, e in enumerate(sub[k]))
                   / len(sol[k]) for k in keys)

    def f_filter(answer_lists, k):
        if "F" in off:
            return answer_lists
        return [[g if M.valid_grid(g) and not viol(k, i, g) else None for i, g in enumerate(a)] for a in answer_lists]

    dsl_res = {}
    if dsl and "D" not in off:
        for r in (json.loads(x) for x in Path(dsl).read_text().splitlines() if x.strip()):
            if r.get("verified") and r["task"] in test:
                dsl_res.setdefault(r["task"], []).append(r["answers"])

    # F audit: no correct LLM answer may be rejected (all logs, all reps)
    f_audit = {"false_rejects": [], "true_rejects": [], "checked_correct": 0, "checked_wrong": 0}
    for name in LLM_LOGS:
        for r in rows(name):
            if not r.get("verified") or r["task"] not in chal:
                continue
            k = r["task"]
            if k not in inv:
                inv[k] = I.infer([(p["input"], p["output"]) for p in chal[k]["train"]])
            for i, g in enumerate(r["answers"]):
                ok = bool((r.get("correct") or [None] * (i + 1))[i])
                f_audit["checked_correct" if ok else "checked_wrong"] += 1
                v = viol(k, i, g) if M.valid_grid(g) else ["invalid"]
                if v:
                    f_audit["false_rejects" if ok else "true_rejects"].append(
                        {"log": name, "task": k, "rep": r.get("rep"), "i": i, "violated": v})

    e3 = rows("E3")
    e6 = rows("E6")
    res = {"off": sorted(off), "active": sorted(active), "dsl": str(dsl) if dsl else None, "cells": {}, "f_audit": f_audit}
    r_lost, r_rescued, cells, cells96 = [], [], [], []
    for s in (1, 2, 3):
        dA = assemble.load_decoder(multi, [(str(OUT_E1 / f"HA-s{s}" / "store"), "")])
        ranked = dA.run_selection_algo()
        base0, baseR = {}, {}
        for k in keys:
            e0, eR = [], []
            for i in range(len(test[k]["test"])):
                full = [np.array(x).tolist() for x in ranked.get(f"{k}_{i}", [])]
                top0 = full[:2]
                topR = top0 if "R" in off else I.stable_filter(full, inv[k], test[k]["test"][i]["input"], active)
                for top, dst in ((top0, e0), (topR, eR)):
                    g = top + [M.PLACEHOLDER] * (2 - len(top))
                    dst.append({"attempt_1": g[0], "attempt_2": g[1]})
                had, has = sol[k][i] in top0, sol[k][i] in topR
                if had and not has:
                    r_lost.append({"seed": s, "task": k, "i": i})
                if has and not had:
                    r_rescued.append({"seed": s, "task": k, "i": i})
            base0[k], baseR[k] = e0, eR
        M.validate(test, base0)
        S_ha, S_base = task_score(base0), task_score(baseR)
        withD, _ = M.merge(test, baseR, dsl_res)
        S_D = task_score(withD)
        for rep, log, cc, acc in ((1, e3, C, cells), (2, e3, C, cells), (1, e6, C96, cells96), (2, e6, C96, cells96)):
            prog = {}
            for r in log:
                if r.get("rep") == rep and r.get("verified") and r["task"] in test:
                    prog.setdefault(r["task"], []).extend(f_filter([r["answers"]], r["task"]))
            results = {k: dsl_res.get(k, []) + prog.get(k, []) for k in set(dsl_res) | set(prog)}
            merged, _ = M.merge(test, baseR, results)
            M.validate(test, merged)
            S_all = task_score(merged)
            # lost-correct check for D+F programs vs the R base
            lost = [(k, i) for k in keys for i, (b, m) in enumerate(zip(baseR[k], merged[k]))
                    if sol[k][i] in (b["attempt_1"], b["attempt_2"]) and sol[k][i] not in (m["attempt_1"], m["attempt_2"])]
            cell = {"seed": s, "rep": rep, "S_ha": round(S_ha, 3), "S_base": round(S_base, 3), "S_D": round(S_D, 3),
                    "S_all": round(S_all, 3), "adj": round(S_D + cc * (S_all - S_D), 3), "lost_correct": lost}
            acc.append(cell)
            res["cells"][f"{'E3' if cc == C else 'E6'}-s{s}-r{rep}"] = cell
    mean = lambda xs, f: round(float(np.mean([x[f] for x in xs])), 3)  # noqa: E731
    res["summary"] = {"HA": mean(cells, "S_ha"), "base_R": mean(cells, "S_base"), "S_D": mean(cells, "S_D"),
                      "raw": mean(cells, "S_all"), "adj": mean(cells, "adj"), "adj_E6_info": mean(cells96, "adj"),
                      "r_lost": r_lost, "r_rescued": r_rescued,
                      "f_false_rejects": len(f_audit["false_rejects"]), "f_true_rejects": len(f_audit["true_rejects"]),
                      "prog_lost_correct": sum(len(c["lost_correct"]) for c in cells)}
    sm = res["summary"]
    sm["pass"] = (sm["adj"] >= 8.253 and not r_lost and sm["f_false_rejects"] == 0 and sm["prog_lost_correct"] == 0)
    res["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / "invariants.py", Path(__file__))}
    Path(dest).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(sm, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    opts = dict(zip(a[2::2], a[3::2]))
    if len(a) < 2 or set(opts) - {"--off", "--dsl"}:
        raise SystemExit(__doc__)
    main(a[0], a[1], tuple(x for x in opts.get("--off", "").split(",") if x), opts.get("--dsl"))
