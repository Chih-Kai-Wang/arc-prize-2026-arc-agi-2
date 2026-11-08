"""Apply the pre-registered E4 -> v4 cap rule (artifacts/verification/arc-agi-2/v13/prereg-e4-cap96k-v1.md) to the
downloaded E4 output: k80 / k96 natural stops among the 10 rep-3 jobs, guards (precision of stops, KV peak,
preemptions, >64k decode speed), projected verified change from cap_yield.py's stratified model -> cap 64k / 80k / 96k.
  uv run --quiet --no-project --with scipy --with numpy python -I runs/arc-agi-2/v13/e4_verdict.py ROOT OUTPUT_DIR [DEST]"""
import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
dest = Path(sys.argv[3]) if len(sys.argv) > 3 else root / "artifacts/verification/arc-agi-2/v13/e4-verdict-v1.json"
sys.argv = [sys.argv[0], str(root)]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(root / "runs/arc-agi-2/v13/cap_yield.py"), run_name="cap_yield")
prereg = json.loads((root / "runs/arc-agi-2/v13/upload-arc-2026-v13-e4-cap96k-v1/preparation.json").read_text())
rows = [json.loads(x) for x in (out / "v13-programs.jsonl").read_text().splitlines() if x.strip()]
rows = [r for r in rows if str(r.get("rep")) == "3"]
n80 = n96 = k80 = k96 = 0
for r in rows:
    ct = int(r.get("completion_tokens") or 0)
    fin = r.get("finish_reason") if r.get("state") == "done" else "aborted"
    if fin == "stop":  # k = verified natural stops PAST 64k only: a fresh-seed stop under 64k would also happen at
        n80 += 1       # the 64k cap, so it is not the cap lever (T3 review 10-16, Opus-7 / Argon-2)
        n96 += 1
        k96 += ct > 64000 and bool(r.get("verified"))
        k80 += 64000 < ct <= 80000 and bool(r.get("verified"))
    elif fin == "length":
        n80 += 1
        n96 += 1
    elif ct > 80000:  # aborted after 80k: "not stopped" by 80k and (conservatively) by 96k (Argon P1-1)
        n80 += 1
        n96 += 1
stops = [r for r in rows if r.get("state") == "done" and r.get("finish_reason") == "stop"]
ver = [r for r in stops if r.get("verified")]
full = [r for r in ver if all(r.get("correct") or [False])]
metrics = [json.loads(x) for x in (out / "v13-metrics.jsonl").read_text().splitlines() if x.strip()] \
    if (out / "v13-metrics.jsonl").exists() else []
kv_peak = max((m.get("vllm:kv_cache_usage_perc", 0) for m in metrics), default=None)
preempt = max((m.get("vllm:num_preemptions_total") or 0 for m in metrics), default=None)  # counter: max (Argon P2-4)


def _speed(min_tokens):
    return [int(r["completion_tokens"]) / (float(r["t_end"]) - float(r.get("t_first") or r["t_start"])) for r in rows
            if r.get("t_end") not in (None, "None") and int(r.get("completion_tokens") or 0) > min_tokens]


speed = _speed(64000) or _speed(20000)  # fallback when no job passed 64k (Argon P2-3)
med_speed = float(np.median(speed)) if speed else None
if med_speed is not None and med_speed < 14:
    g["plan_strat"].__globals__["SPEED"] = med_speed  # run_path returns a copy; patch the functions' namespace
# KV: E3 fit usage = 0.0119 * seqs + 7.663e-7 * in-flight tokens (R2 0.993) -> E4 all 10 at 96k predicted 0.906.
# Prereg item 3: KV peak >= 0.95 or preemptions -> recompute the v4 concurrency from the measurement (not "go 64k");
# preemption delays streams but does not change what they sample, so the stop counts stay valid.
KV_A, KV_B, E4_PRED_PEAK = 0.0119, 7.663e-7, 0.906
kv_scale = max(1.0, (kv_peak or 0) / E4_PRED_PEAK) if (kv_peak or 0) >= 0.95 or preempt else 1.0


def conc_for(cap, formula):  # any preemption: one stream fewer on top of the KV rescale (T3 review, Argon-8)
    return max(8, min(formula, int(0.92 / (kv_scale * (KV_A + KV_B * (cap + 6000))))) - (1 if preempt else 0))


c80, c96 = conc_for(80000, 11), conc_for(96000, 10)
base = g["plan_strat"](64000, 13, 0.0)["verified"]
q80 = k80 / n80 if n80 else 0.0
q96 = k96 / n96 if n96 else 0.0
d80 = g["plan_strat"](80000, c80, q80)["verified"] - base
d96 = g["plan_strat"](96000, c96, q96)["verified"] - base
base_f = g["plan_strat"](64000, 13, 0.0, reach_frozen=True)["verified"]
d80_f = g["plan_strat"](80000, c80, q80, reach_frozen=True)["verified"] - base_f
d96_f = g["plan_strat"](96000, c96, q96, reach_frozen=True)["verified"] - base_f
guards = {"stop_precision_ok": (len(full) / len(ver) >= 0.5) if ver else True}
if n80 < 8 and n96 < 8:  # too few finished jobs (e.g. slow decode / deadline aborts): no answer, not "keep 64k"
    cap = "inconclusive"
elif not all(guards.values()):
    cap = 64000
elif k96 >= 3 and n96 >= 8 and d96 > d80 and d96_f >= 0:
    cap = 96000
elif k80 >= 2 and n80 >= 8 and d80_f >= 0:
    cap = 80000
else:
    cap = 64000
res = {"kernel": prereg["kernel"], "jobs_seen": len(rows), "n80": n80, "k80": k80, "n96": n96, "k96": k96,
       "q80": round(q80, 3), "q96": round(q96, 3), "proj_d_verified": {f"80k_c{c80}": round(d80, 1), f"96k_c{c96}": round(d96, 1)},
       "proj_d_verified_reach_frozen": {f"80k_c{c80}": round(d80_f, 1), f"96k_c{c96}": round(d96_f, 1)},
       "stops": len(stops), "stops_verified": len(ver), "stops_full_correct": len(full), "kv_peak": kv_peak,
       "preemptions": preempt, "kv_scale": round(kv_scale, 3), "v4_conc": {"80k": c80, "96k": c96},
       "median_tok_s_over_64k": med_speed, "guards": guards, "v4_cap": cap,
       "rule": "prereg-e4-cap96k-v1.md (+ amendments 1-3): k = verified natural stops past 64k; inconclusive if n80<8 and n96<8; 96k if k96>=3, n96>=8, "
               "d96>d80, d96_frozen>=0; else 80k if k80>=2, n80>=8, d80_frozen>=0; else 64k; stop-precision guard "
               "first; KV>=0.95 or preemption -> v4 conc recomputed from the measured peak"}
dest.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
