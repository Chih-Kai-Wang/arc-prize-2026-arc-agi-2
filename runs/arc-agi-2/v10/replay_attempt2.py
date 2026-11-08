"""Replay attempt_2 rules on stored Colab candidates (prereg: artifacts/verification/arc-agi-2/v10/prereg-attempt2-replay-v1.json).

Usage: .venv/bin/python runs/arc-agi-2/v10/replay_attempt2.py A_DIR PAIR_DIR   (A_DIR/out_a, PAIR_DIR/out_b360, PAIR_DIR/out_b1200)
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/v9/src"))
from arc_loader import ArcDataset  # noqa: E402
import assemble  # noqa: E402

DATA = ROOT / "data/arc-agi-2/official-v1"
KEYS = json.loads((ROOT / "runs/arc-agi-2/colab-pair-v1/keys_b.json").read_text())


def v10(sel_a, sel_b, v8):
    out = {}
    for bk, pair in v8.items():
        first = pair[0] if pair else None
        b_top = (sel_b.get(bk) or [None])[0]
        if first is not None and b_top is not None and not np.array_equal(b_top, first):
            out[bk] = [first, b_top]
        else:
            out[bk] = pair
    return out


def hits(selected, multi):
    return {bk: any(np.array_equal(g, multi.replies[bk][0]) for g in selected.get(bk, [])[:2]) for bk in multi.keys}


def main(a_dir, pair_dir):
    data = ArcDataset.from_file(str(DATA / "arc-agi_evaluation_challenges.json")).change_keys(KEYS, keep_flags=True)
    sol = json.loads((DATA / "arc-agi_evaluation_solutions.json").read_text())
    data.replies = {k: sol[k] for k in KEYS}
    multi = data.split_multi_replies()
    n_tests = {k: sum(1 for bk in multi.keys if bk.rsplit("_", 1)[0] == k) for k in KEYS}
    for store in [Path(a_dir) / "out_a"] + [Path(pair_dir) / f"out_{b}" for b in ("b360", "b1200")]:
        if not store.is_dir() or not any(store.iterdir()):
            raise SystemExit(f"missing or empty store: {store}")
    A = [(str(Path(a_dir) / "out_a"), "")]
    sel_a = assemble.load_decoder(multi, A).run_selection_algo()
    result = {"prereg": "artifacts/verification/arc-agi-2/v10/prereg-attempt2-replay-v1.json", "variants": {}}
    for b in ("b360", "b1200"):
        B = [(str(Path(pair_dir) / f"out_{b}"), ".b")]
        sel_b = assemble.load_decoder(multi, B).run_selection_algo()
        sel_p = assemble.load_decoder(multi, A + B).run_selection_algo()
        v8, _ = assemble.keep_a_top1(sel_a, sel_p)
        cand = v10(sel_a, sel_b, v8)
        h8, h10 = hits(v8, multi), hits(cand, multi)
        score = lambda h: round(sum(h[bk] / n_tests[bk.rsplit("_", 1)[0]] for bk in h), 3)
        result["variants"][b] = {"pass2_v8_current": score(h8), "pass2_v10_candidate": score(h10),
                                 "attempt2_changed": sum(1 for bk in v8 if len(v8[bk]) != len(cand[bk]) or any(
                                     not np.array_equal(x, y) for x, y in zip(v8[bk], cand[bk]))),
                                 "gained": [bk for bk in h8 if h10[bk] and not h8[bk]],
                                 "lost": [bk for bk in h8 if h8[bk] and not h10[bk]]}
    v = result["variants"]
    no_loss = all(not x["lost"] and x["pass2_v10_candidate"] >= x["pass2_v8_current"] for x in v.values())
    gain = any(x["gained"] for x in v.values())
    result["decision"] = ("adopt-candidate" if no_loss and gain else "no-evidence (tie)" if no_loss else "reject (loss)")
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
