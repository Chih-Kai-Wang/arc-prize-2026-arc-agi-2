"""C2 measurement (prereg artifacts/verification/arc-agi-2/v13/prereg-c2-effort-repair-v1.md section 2 + amendment 3).
Dev-run rows with t_end <= gen_start + 5,825 s (all rows if the generation phase was shorter), merged onto E1 HA top-2
(seeds 1-3) with v13_merge.merge -> adj = mean over seeds of the keys30 score. v2 per task = HA + 0.65 x mean E3
rep-1/rep-2 task gain. Paired bootstrap over the 30 tasks -> P(delta > 0). Also the same cut on v2's own dev run.
  .venv/bin/python runs/arc-agi-2/v13/c2_measure.py OUT_DIR DEST.json     (OUT_DIR holds v13-programs.jsonl + v13-llm.json)"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/gcp-e1"))
import analyze_e1 as E  # noqa: E402
from analyze_e1 import ArcDataset, analyze, assemble  # noqa: E402
sys.path.insert(0, str(HERE / "src"))
import v13_merge as M  # noqa: E402

T_STAR, C, V2_ADJ = 5825.0, 0.65, 7.753
RUNS = ROOT / "runs/arc-agi-2/v6"
E3 = RUNS / "output-arc-2026-v13-e3-seeds-v1/v13-programs.jsonl"
V2_DEV = RUNS / "output-arc-2026-v13-progsynth-v2"


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


class Base:
    def __init__(self):
        self.keys = json.loads((E.HERE / "build/e1-inputs-v1.manifest.json").read_text())["keys30"]
        chal = json.loads((analyze.DATA / "arc-agi_evaluation_challenges.json").read_text())
        self.sol = json.loads((analyze.DATA / "arc-agi_evaluation_solutions.json").read_text())
        data = ArcDataset.from_file(str(analyze.DATA / "arc-agi_evaluation_challenges.json")).change_keys(self.keys, keep_flags=True)
        data.replies = {k: self.sol[k] for k in self.keys}
        multi = data.split_multi_replies()
        self.test = {k: chal[k] for k in self.keys}
        self.base = {}
        for s in (1, 2, 3):
            sel = analyze.select(assemble.load_decoder(multi, [(str(ROOT / f"runs/arc-agi-2/gcp-e1/out/e1/HA-s{s}/store"), "")])
                                 .run_selection_algo())
            b = {}
            for k in self.keys:
                ent = []
                for i in range(len(self.test[k]["test"])):
                    g = [np.array(x).tolist() for x in sel.get(f"{k}_{i}", [])[:2]]
                    g += [M.PLACEHOLDER] * (2 - len(g))
                    ent.append({"attempt_1": g[0], "attempt_2": g[1]})
                b[k] = ent
            self.base[s] = b

    def per_task(self, sub):
        return np.array([sum(any(g == self.sol[k][i] for g in (e["attempt_1"], e["attempt_2"])) for i, e in enumerate(sub[k]))
                         / len(self.sol[k]) for k in self.keys])

    def scores(self, results):
        """seed -> per-task score array, for program answers `results` (task -> list of answer lists)."""
        res = {k: v for k, v in results.items() if k in self.test}
        return {s: self.per_task(M.merge(self.test, self.base[s], res)[0]) for s in (1, 2, 3)}


def verified(rs):
    out = {}
    for r in rs:
        if r.get("verified") and isinstance(r.get("answers"), list):
            out.setdefault(r["task"], []).append(r["answers"])
    return out


def cut(out_dir, horizon=T_STAR):
    llm = json.loads((Path(out_dir) / "v13-llm.json").read_text())
    rs = rows(Path(out_dir) / "v13-programs.jsonl")
    g0 = float(llm["gen_start"])
    ends = [float(r["t_end"]) for r in rs if r.get("t_end") is not None]
    gen = (max(ends) - g0) if ends else 0.0
    kept = [r for r in rs if r.get("t_end") is not None and float(r["t_end"]) <= g0 + horizon]
    return kept, {"gen_start": g0, "gen_seconds_observed": round(gen, 1), "rows": len(rs), "rows_kept": len(kept),
                  "full_window": gen >= T_STAR, "horizon": horizon}


def main(out_dir, dest):
    B = Base()
    ha = {s: B.per_task(B.base[s]) for s in (1, 2, 3)}
    e3 = rows(E3)
    gain = {s: np.mean([B.scores(verified([r for r in e3 if r.get("rep") == rep]))[s] - ha[s] for rep in (1, 2)], axis=0)
            for s in (1, 2, 3)}
    v2_task = np.mean([ha[s] + C * gain[s] for s in (1, 2, 3)], axis=0)
    res = {"t_star": T_STAR, "v2_adj_check": round(float(v2_task.sum()), 3)}
    horizon = T_STAR
    for name, d in (("c2", out_dir), ("v2_dev", V2_DEV)):
        kept, info = cut(d, horizon)
        if name == "c2":  # amendment 7: v2's dev run is cut at the same horizon (C2's commit is 200 s shorter)
            horizon = min(T_STAR, info["gen_seconds_observed"])
        sc = B.scores(verified(kept))
        task = np.mean([sc[s] for s in (1, 2, 3)], axis=0)
        info.update(adj=round(float(task.sum()), 3), per_seed={s: round(float(sc[s].sum()), 3) for s in (1, 2, 3)},
                    verified_rows_kept=sum(bool(r.get("verified")) for r in kept))
        if not info["full_window"] and info["gen_seconds_observed"] > 0:  # information only (amendment 3)
            ha_m = float(np.mean([ha[s].sum() for s in (1, 2, 3)]))
            info["scaled_info_only"] = round(ha_m + T_STAR / info["gen_seconds_observed"] * (info["adj"] - ha_m), 3)
        res[name] = info
        if name == "c2":
            delta = task - v2_task
            rng = np.random.default_rng(20261019)
            idx = rng.integers(0, len(delta), size=(10000, len(delta)))
            boot = delta[idx].sum(axis=1)
            res["delta_mean"] = round(float(delta.sum()), 3)
            res["p_delta_gt_0"] = round(float((boot > 0).mean()), 4)
            res["per_task_delta"] = {k: round(float(x), 3) for k, x in zip(B.keys, delta) if abs(x) > 1e-9}
    res["slot1_offline_criteria_met"] = (res["delta_mean"] >= 0.5 and res["p_delta_gt_0"] >= 0.8
                                         and res["c2"]["adj"] >= res["v2_dev"]["adj"] + 0.5)  # amendment 5
    Path(dest).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_task_delta"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
