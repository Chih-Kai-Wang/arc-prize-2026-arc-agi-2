"""Per-task pass@2 (F, H_pooled) and oracle per seed from E1, vs P0b program-channel solved tasks. Usage: python e1_pertask.py ROOT OUT"""
import json, sys
from pathlib import Path
import numpy as np
root, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root / "runs/arc-agi-2/gcp-e1"))
import analyze_e1 as E
from analyze_e1 import assemble, analyze, ArcDataset
keys = json.loads((E.HERE / "build/e1-inputs-v1.manifest.json").read_text())["keys30"]
data = ArcDataset.from_file(str(analyze.DATA / "arc-agi_evaluation_challenges.json")).change_keys(keys, keep_flags=True)
sol = json.loads((analyze.DATA / "arc-agi_evaluation_solutions.json").read_text())
data.replies = {k: sol[k] for k in keys}
multi = data.split_multi_replies()
def per_task(selected):
    r = {}
    for k in keys:
        bks = [b for b in multi.keys if b.rsplit("_", 1)[0] == k]
        r[k] = sum(any(np.array_equal(g, multi.replies[b][0]) for g in selected.get(b, [])[:2]) for b in bks) / len(bks)
    return r
def oracle_task(dec):
    r = {}
    for k in keys:
        bks = [b for b in multi.keys if b.rsplit("_", 1)[0] == k]
        r[k] = sum(any(np.array_equal(s["solution"], multi.replies[b][0]) for s in dec.decoded_results.get(b, {}).values()) for b in bks) / len(bks)
    return r
res = {}
for s in (1, 2, 3):
    runs = {r: out / f"{r}-s{s}" for r in ("F", "HA", "HB")}
    dF = assemble.load_decoder(multi, [(str(runs["F"] / "store"), "")])
    dA = assemble.load_decoder(multi, [(str(runs["HA"] / "store"), "")])
    dAB = assemble.load_decoder(multi, [(str(runs["HA"] / "store"), ""), (str(runs["HB"] / "store"), ".b")])
    sF, sA, sAB = dF.run_selection_algo(), dA.run_selection_algo(), dAB.run_selection_algo()
    pooled, _ = assemble.keep_a_top1(sA, sAB)
    res[s] = {"F": per_task(analyze.select(sF)), "H": per_task(pooled), "F_oracle": oracle_task(dF)}
p0b = {}
for tag in ("q38", "q36a3b"):
    for l in open(root / f"runs/arc-agi-2/p0b-colab/out/out/{tag}-calls.jsonl"):
        r = json.loads(l)
        if r.get("kind") == "program" and r.get("verified"):
            p0b.setdefault(r["task"], set()).add(tag)
p0b_correct = {}
for tag in ("q38", "q36a3b"):
    summ = json.load(open(root / f"runs/arc-agi-2/p0b-colab/out/out/{tag}-summary.json"))
print("task F_s1 F_s2 F_s3 | H_s1 H_s2 H_s3 | Foracle(any seed) | p0b_verified")
rows = {}
for k in keys:
    f = [res[s]["F"][k] for s in (1, 2, 3)]; h = [res[s]["H"][k] for s in (1, 2, 3)]
    o = max(res[s]["F_oracle"][k] for s in (1, 2, 3))
    rows[k] = {"F": f, "H": h, "F_oracle_any": o, "p0b": sorted(p0b.get(k, []))}
    if any(f) or any(h) or k in p0b or o:
        print(k, *[round(x, 2) for x in f], "|", *[round(x, 2) for x in h], "|", round(o, 2), "|", ",".join(sorted(p0b.get(k, []))))
json.dump(rows, open(sys.argv[3], "w"), indent=1)
