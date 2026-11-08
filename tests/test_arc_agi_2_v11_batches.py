"""v11 decode batching: the author's fixed 16-view pairing is reproduced exactly (checked against the v9 solver's own
code), and other view counts (half-cost: 8 views) get non-empty, equal-length batches covering every view once.
  .venv/bin/python -m unittest tests.test_arc_agi_2_v11_batches -v
"""
from collections import defaultdict
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/v11/src"))
import arc_v6_config as cfg  # noqa: E402

GEOMS = ["", "rot90.", "rot90.rot90.", "rot90.rot90.rot90.", "transpose.", "transpose.rot90.",
         "transpose.rot90.rot90.", "transpose.rot90.rot90.rot90."]


def v9_batches(test_id_to_subkeys):
    """Run the v9 solver's batching block verbatim."""
    src = (ROOT / "runs/arc-agi-2/v9/src/arc_solver.py").read_text()
    start = src.index("        batches = []\n        for test_id, subkeys in test_id_to_subkeys.items():")
    end_marker = "            for offset in [10, 14]:\n                batch.extend(subkeys[offset:offset+2])\n            batches.append(batch)\n"
    block = src[start:src.index(end_marker, start) + len(end_marker)]
    ns = {"test_id_to_subkeys": test_id_to_subkeys}
    exec("\n".join(line[8:] for line in block.splitlines()), ns)
    return ns["batches"]


def views(rng, task, tests, per_geom):
    out = defaultdict(list)
    for t in range(tests):
        keys = [f"{task}_{t}.{g}permute{''.join(rng.sample('0123456789', 10))}.ex{rng.randint(0, 99)}"
                for g in GEOMS for _ in range(per_geom)]
        for k in sorted(keys):
            out[str(t)].append(k)
    return out


class Batches(unittest.TestCase):
    def test_16_views_match_v9_exactly(self):
        rng = random.Random(7)
        for tests in (1, 2, 3):
            m = views(rng, "%08x" % rng.getrandbits(32), tests, 2)
            self.assertTrue(all(len(v) == 16 for v in m.values()))
            self.assertEqual(cfg.decode_batches(m, lambda sk: 0), v9_batches(m))

    def test_8_views_equal_length_cover_once(self):
        rng = random.Random(8)
        for h, w in ((3, 5), (4, 4), (30, 1)):
            for tests in (1, 3):
                m = views(rng, "%08x" % rng.getrandbits(32), tests, 1)
                L = lambda sk: (h if (sk.count("rot90") % 2 == 1) == ("transpose" in sk) else w)
                b = cfg.decode_batches(m, L)
                self.assertTrue(all(1 <= len(x) <= 4 for x in b))
                self.assertTrue(all(len({L(sk) for sk in x}) == 1 for x in b))
                flat = [sk for x in b for sk in x]
                self.assertEqual(sorted(flat), sorted(sk for v in m.values() for sk in v))
                for t, subkeys in m.items():   # every test input gets at least one batch, batches stay within a test
                    self.assertTrue(any(x[0] in subkeys for x in b))
                self.assertTrue(all(len({sk.split(".")[0] for sk in x}) == 1 for x in b))

    def test_partial_views_never_empty(self):
        rng = random.Random(9)
        m = views(rng, "abcdef01", 2, 2)
        m["1"] = m["1"][:11]                      # e.g. cut_to_len dropped some views
        b = cfg.decode_batches(m, lambda sk: sk.count("rot90") % 2)
        self.assertTrue(all(x for x in b))
        self.assertEqual(sum(len(x) for x in b), 16 + 11)


if __name__ == "__main__":
    unittest.main()
