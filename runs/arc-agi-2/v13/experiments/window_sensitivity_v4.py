"""v4-c96 pre-submit review P1-1 (Opus 10-17): sensitivity of the stratified cap model to the hidden-run LLM window.
Read-only: exec's cap_yield.py's data loading + stratified fit (no json writes) and re-evaluates plan_strat for windows
1.5-6 h, q from E4 (q80 0.2 / q96 0.4) and break-even probes, with and without reach_frozen.
  uv run --quiet --no-project --with scipy --with numpy python -I runs/arc-agi-2/v13/experiments/window_sensitivity_v4.py ROOT"""
import json
import sys
from pathlib import Path

root = sys.argv[1]
src = (Path(root) / "runs/arc-agi-2/v13/cap_yield.py").read_text()
head = src.split("fits = {", 1)[0]                                    # data load, fits, constants, plan()
strat = src.split("# --- stratified model", 1)[1].split("strat = {}", 1)[0]  # stratified fit + plan_strat()
ns = {"__name__": "cap_yield_ro"}
sys.argv = [sys.argv[0], root]
exec(compile(head + "\n# --- stratified model" + strat, "cap_yield_ro", "exec"), ns)

out = {"source": "cap_yield.py (exec'd read-only)", "speed": ns["SPEED"], "rows": []}
for hours in (1.5, 2, 3, 4, 5, 6):
    ns["W"] = hours * 3600
    for frozen in (True, False):
        base = ns["plan_strat"](64000, 13, 0.0, frozen)
        row = {"window_h": hours, "reach_frozen": frozen, "v64_c13": base["verified"]}
        for cap, conc, qs in ((80000, 11, (0.2,)), (96000, 10, (0.2, 0.29, 0.4))):
            for q in qs:
                p = ns["plan_strat"](cap, conc, q, frozen)
                row[f"d_{cap // 1000}k_c{conc}_q{q}"] = round(p["verified"] - base["verified"], 2)
        out["rows"].append(row)
for r in out["rows"]:
    print(r)
dest = Path(root) / "artifacts/verification/arc-agi-2/v13/window-sensitivity-v4-c96.json"
dest.write_text(json.dumps(out, indent=1))
print("wrote", dest)
