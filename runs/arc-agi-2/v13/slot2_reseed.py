"""Slot-2 reseed vs v8 (zero GPU; rule pre-registered in artifacts/verification/arc-agi-2/v13/design-1024-paper-screen-v1.md
before this ran). Per-task keys30 cells: v13 = HA seed s + E3 program rep r per-task delta (64k = v2 config), F = E1 F seed s.
One simulated private set = 120 tasks drawn from keys30 with replacement; each run's program gains apply with prob c (own
mask). A = v13 cell (s,r); reseed B = cell (s'!=s, r'!=r); v8 B = F seed s (shared pass-A seeds, like the real pair).
Gain = E[(B-A)+] in pp. Outer bootstrap re-draws the 30 keys to show estimate uncertainty.
  .venv/bin/python runs/arc-agi-2/v13/slot2_reseed.py artifacts/verification/arc-agi-2/v13/slot2-reseed-v1.json"""
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


def per_task(selected):  # same as slot2_pertask.per_task
    r = {}
    for k in keys:
        bks = [b for b in multi.keys if b.rsplit("_", 1)[0] == k]
        r[k] = sum(any(np.array_equal(g, multi.replies[b][0]) for g in selected.get(b, [])[:2]) for b in bks) / len(bks)
    return r


HA = {s: per_task(analyze.select(assemble.load_decoder(multi, [(str(out / f"HA-s{s}" / "store"), "")]).run_selection_algo()))
      for s in (1, 2, 3)}
reps = {r: json.loads((V / f"v13/replay-e3-rep{r}.json").read_text()) for r in (1, 2)}
for r, rp in reps.items():
    for s in (1, 2, 3):
        assert abs(sum(HA[s].values()) - rp["seeds"][str(s)]["before"]) < 1e-3, (r, s)
base = np.array([[HA[s][k] for k in keys] for s in (1, 2, 3)])                       # [s, task]
delta = np.array([[[reps[r]["seeds"][str(s)]["per_task_delta"].get(k, 0.0) for k in keys] for s in (1, 2, 3)]
                  for r in (1, 2)])                                                   # [r, s, task]
F = np.array([[pert[k]["F"][s - 1] for k in keys] for s in (1, 2, 3)])               # [s, task]
# the cells must reproduce the recorded totals (slot2-pertask-v1.json)
ref = json.loads((V / "v13/slot2-pertask-v1.json").read_text())["totals"]
assert np.allclose(F.sum(1), ref["F_v8"], atol=1e-2), F.sum(1)
for r in (1, 2):
    for s in (1, 2, 3):
        assert abs((base[s - 1] + delta[r - 1, s - 1]).sum() - ref["v13_cells"][f"rep{r}-s{s}"]) < 1e-2, (r, s)

N_TASKS, INNER, OUTER, LINE, PUB_A = 120, 4000, 300, 32.36, 32.22


def simulate(idx, c, rng, inner=INNER):
    """idx: the keys30 indices standing for the task population. Returns gains and P(max>=LINE) for both pairs."""
    t = idx[rng.integers(0, len(idx), (inner, N_TASKS))]                              # [inner, 120]
    s1 = rng.integers(0, 3, inner)
    r1 = rng.integers(0, 2, inner)
    s2 = (s1 + rng.integers(1, 3, inner)) % 3                                         # s' != s
    r2 = 1 - r1                                                                        # r' != r
    mA, mB = rng.random((inner, N_TASKS)) < c, rng.random((inner, N_TASKS)) < c
    A = base[s1[:, None], t] + mA * delta[r1[:, None], s1[:, None], t]
    Bre = base[s2[:, None], t] + mB * delta[r2[:, None], s2[:, None], t]
    Bv8 = F[s1[:, None], t]
    a, bre, bv8 = (x.mean(1) * 100 for x in (A, Bre, Bv8))                           # pp on the simulated set
    g_re, g_v8 = np.maximum(bre - a, 0).mean(), np.maximum(bv8 - a, 0).mean()
    shift = PUB_A - a.mean()
    p = {name: float((np.maximum(a, b) + shift >= LINE).mean()) for name, b in (("reseed", bre), ("v8", bv8))}
    p["A_alone"] = float((a + shift >= LINE).mean())
    return {"gain_reseed": float(g_re), "gain_v8": float(g_v8), "d_v8_mean_pp": float((a - bv8).mean()),
            "sd_reseed_pp": float((bre - a).std()), "sd_v8_pp": float((bv8 - a).std()), "p_max_ge_line": p}


rng = np.random.default_rng(20261018)
allidx = np.arange(len(keys))
res = {"keys": len(keys), "n_tasks": N_TASKS, "inner": INNER, "outer": OUTER, "line": LINE, "public_A": PUB_A,
       "rule": "switch-candidate only if at c=0.65 median(Δ=gain_reseed-gain_v8) >= +0.2pp and P(Δ>0) >= 0.8, "
               "and median Δ > 0 at c=0.5 and c=0.8; else keep v8 (design-1024-paper-screen-v1.md)",
       "by_c": {}}
for c in (0.5, 0.65, 0.8):
    point = simulate(allidx, c, rng, inner=20000)
    boot = [simulate(allidx[rng.integers(0, len(keys), len(keys))], c, rng) for _ in range(OUTER)]
    d = np.array([b["gain_reseed"] - b["gain_v8"] for b in boot])
    res["by_c"][str(c)] = {"point": point, "delta_median": round(float(np.median(d)), 3),
                           "delta_q10_q90": [round(float(np.quantile(d, q)), 3) for q in (0.1, 0.9)],
                           "p_delta_pos": round(float((d > 0).mean()), 3)}
r65 = res["by_c"]["0.65"]
res["pass"] = bool(r65["delta_median"] >= 0.2 and r65["p_delta_pos"] >= 0.8
                   and all(res["by_c"][c]["delta_median"] > 0 for c in ("0.5", "0.8")))
Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
