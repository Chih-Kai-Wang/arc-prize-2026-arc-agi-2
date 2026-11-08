"""T2 P1 (v5 pre-launch, 10-18): can the v5 run change the slot-2 decision? Same per-task cells and simulation as
slot2_reseed.py, but candidate B (v5 = reseeded v13 at conc 16) gets coverage c_B = min(1, r * c_A), where c_A is
v13 v2's hidden coverage scenario and r the throughput ratio the commit run measures. Zero GPU.
  .venv/bin/python runs/arc-agi-2/v13/slot2_v5_whatif.py artifacts/verification/arc-agi-2/v13/slot2-v5-whatif-v1.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.argv, OUT = sys.argv[:1] + ["/dev/null"], sys.argv[1]
HERE = Path(__file__).resolve().parent
src = (HERE / "slot2_reseed.py").read_text()
ns = {"__file__": str(HERE / "slot2_reseed.py"), "__name__": "slot2_reseed_lib"}
exec(compile(src.split("\nrng = np.random.default_rng(20261018)")[0], "slot2_reseed.py", "exec"), ns)  # loaders only
base, delta, F, keys, N = ns["base"], ns["delta"], ns["F"], ns["keys"], ns["N_TASKS"]


def sim(idx, cA, cB, rng, inner):
    t = idx[rng.integers(0, len(idx), (inner, N))]
    s1, r1 = rng.integers(0, 3, inner), rng.integers(0, 2, inner)
    s2, r2 = (s1 + rng.integers(1, 3, inner)) % 3, 1 - r1
    mA, mB = rng.random((inner, N)) < cA, rng.random((inner, N)) < cB
    a = (base[s1[:, None], t] + mA * delta[r1[:, None], s1[:, None], t]).mean(1) * 100
    b5 = (base[s2[:, None], t] + mB * delta[r2[:, None], s2[:, None], t]).mean(1) * 100
    b8 = F[s1[:, None], t].mean(1) * 100
    return float(np.maximum(b5 - a, 0).mean() - np.maximum(b8 - a, 0).mean())


rng = np.random.default_rng(20261018)
res = {"by": {}}
for cA in (0.5, 0.65, 0.8):
    for r in (1.0, 1.1, 1.2, 1.3):
        cB = min(1.0, r * cA)
        point = sim(np.arange(len(keys)), cA, cB, rng, 20000)
        boot = np.array([sim(np.arange(len(keys))[rng.integers(0, len(keys), len(keys))], cA, cB, rng, 3000)
                         for _ in range(200)])
        res["by"][f"cA={cA},r={r}"] = {"cB": round(cB, 3), "delta_point": round(point, 3),
                                       "delta_median": round(float(np.median(boot)), 3),
                                       "p_delta_pos": round(float((boot > 0).mean()), 3)}
        print(cA, r, res["by"][f"cA={cA},r={r}"], flush=True)
Path(OUT).write_text(json.dumps(res, indent=1) + "\n")
