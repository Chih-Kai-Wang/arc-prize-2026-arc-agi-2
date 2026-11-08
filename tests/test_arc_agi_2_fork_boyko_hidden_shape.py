"""Hidden-data-shape test for the Boyko V22 fork (runs/arc-agi-2/fork-boyko, promote off, commit budget trimmed).

No GPU, no model. The notebook's real finalizer program (the r'''...''' string of its last code cell, run in a fresh
process exactly as the notebook does, only /kaggle paths redirected) runs on the fork's real extracted modules over
synthetic test files with fresh random IDs, 7 and 13 tasks, shuffled order, 1-2 test inputs and varied grid sizes, with
pass-A and pass-B decoded files for some inputs and none for others (identity fallback path).
Run:  .venv/bin/python -m unittest tests.test_arc_agi_2_fork_boyko_hidden_shape -v
Evidence JSON:  .venv/bin/python tests/test_arc_agi_2_fork_boyko_hidden_shape.py --write-evidence
"""
import ast
import bz2
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "runs/arc-agi-2/fork-boyko"
SRC = FORK / "src"
NOTEBOOK = FORK / "upload-arc-2026-fork-boyko-v1/arc-2026-fork-boyko-v1.ipynb"
EVIDENCE = ROOT / "artifacts/verification/arc-agi-2/fork-boyko/hidden-shape-test-v1.json"
DEBUG_IDS = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]


def code_cells():
    return ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]


def finalizer_source():
    cell = [c for c in code_cells() if "finalizer_code = r'''" in c][-1]
    for node in ast.walk(ast.parse(cell)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "finalizer_code":
            return node.value.value
    raise AssertionError("finalizer_code not found")


def rand_grid(rng, rows, cols):
    return [[rng.randrange(10) for _ in range(cols)] for _ in range(rows)]


def make_challenges(rng, n_tasks, taken):
    tasks = {}
    while len(tasks) < n_tasks:
        tid = "".join(rng.choice("0123456789abcdef") for _ in range(8))
        if tid in taken or tid in tasks:
            continue
        tasks[tid] = {"train": [{"input": rand_grid(rng, rng.randint(1, 30), rng.randint(1, 30)),
                                 "output": rand_grid(rng, rng.randint(1, 30), rng.randint(1, 30))}
                                for _ in range(rng.randint(2, 4))],
                      "test": [{"input": rand_grid(rng, rng.randint(1, 30), rng.randint(1, 30))}
                               for _ in range(rng.choice([1, 2]))]}
    items = list(tasks.items())
    rng.shuffle(items)
    return dict(items)


def write_decoded(directory, key, solutions, aug_means):
    rows = [{"beam_score": 0.1 * (i + 1), "score_aug": [m] * 8, "solution": np.array(s, dtype=int)}
            for i, (s, m) in enumerate(zip(solutions, aug_means))]
    with bz2.BZ2File(os.path.join(directory, key), "w") as f:
        pickle.dump(rows, f)


def run_finalizer(challenges, plan_a, plan_b, rerun=True, extra_env=None, expert_stub=False):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        comp, work, a_dir, b_dir, stub = (tmp / n for n in ("comp", "working", "outA", "outB", "stub"))
        for d in (comp, work, stub):
            d.mkdir()
        (stub / "transformers.py").write_text("class AutoTokenizer: pass\n")  # loader imports it for a type only
        if expert_stub:  # shadows src/astra_bridge.py: overwrites every attempt_2 only when the finalizer passes promote=True
            (stub / "astra_bridge.py").write_text(
                "import copy\n"
                "def run_expert(queries, baseline, promote, time_budget_seconds):\n"
                "    sub = copy.deepcopy(baseline)\n"
                "    if promote:\n"
                "        for v in sub.values():\n"
                "            for e in v:\n"
                "                e['attempt_2'] = [[7]]\n"
                "    return sub, {k: {'status': 'promoted_to_attempt_2' if promote else 'shadow_only'} for k in sub}\n")
        (comp / "arc-agi_test_challenges.json").write_text(json.dumps(challenges))
        for d, plan in ((a_dir, plan_a), (b_dir, plan_b)):
            if plan is not None:
                d.mkdir()
                for key, (sols, means) in plan.items():
                    write_decoded(d, key, sols, means)
        code = (finalizer_source().replace("/kaggle/input/competitions/arc-prize-2026-arc-agi-2", str(comp))
                .replace("/kaggle/inference_outputs_b", str(b_dir)).replace("/kaggle/inference_outputs", str(a_dir))
                .replace("/kaggle/working", str(work)))
        assert "/kaggle/" not in code.replace("'/kaggle/input/arc-prize-2026-arc-agi-2'", "")
        env = {k: v for k, v in os.environ.items() if k != "KAGGLE_IS_COMPETITION_RERUN"}
        env["PYTHONPATH"] = f"{stub}{os.pathsep}{SRC}"
        env.pop("ARC_ASTRA_PROMOTE", None)
        if rerun:
            env["KAGGLE_IS_COMPETITION_RERUN"] = "1"
        env.update(extra_env or {})
        r = subprocess.run([sys.executable, "-c", code], cwd=work, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        return json.loads((work / "submission.json").read_text()), r.stdout


def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and len(r) > 0 for r in g)
            and len({len(r) for r in g}) == 1 and all(type(x) is int and 0 <= x <= 9 for r in g for x in r))


class BoykoForkHiddenShape(unittest.TestCase):
    def assert_valid(self, challenges, sub):
        self.assertEqual(set(sub), set(challenges))
        for tid, task in challenges.items():
            self.assertEqual(len(sub[tid]), len(task["test"]))
            for e in sub[tid]:
                self.assertEqual(set(e), {"attempt_1", "attempt_2"})
                self.assertTrue(valid_grid(e["attempt_1"]) and valid_grid(e["attempt_2"]), tid)

    def check(self, n, seed):
        rng = random.Random(seed)
        challenges = make_challenges(rng, n, set(DEBUG_IDS))
        inputs = [(t, i) for t, task in challenges.items() for i in range(len(task["test"]))]
        dec = set(rng.sample(inputs, len(inputs) // 2))
        plan_a, plan_b, exp = {}, {}, {}
        for t, i in inputs:
            if (t, i) not in dec:
                continue
            best, second = (rand_grid(rng, rng.randint(1, 30), rng.randint(1, 30)) for _ in range(2))
            plan_a[f"{t}_{i}.permute0123456789"] = ([best, second], [0.2, 0.9])
            if rng.random() < 0.5:
                plan_b[f"{t}_{i}.rot90.permute9876543210"] = ([best], [0.3])
            exp[(t, i)] = best
        sub, out = run_finalizer(challenges, plan_a, plan_b)
        self.assert_valid(challenges, sub)
        for t, i in inputs:
            e = sub[t][i]
            if (t, i) in exp:
                self.assertTrue(e["attempt_1"] == exp[(t, i)] or e["attempt_2"] == exp[(t, i)])
            else:  # identity fallback for undecoded inputs
                self.assertEqual(e["attempt_1"], challenges[t]["test"][i]["input"])
        self.assertIn("promote=False", out)
        return dec, inputs

    def test_hidden_shape_7_tasks(self):
        dec, inputs = self.check(7, 20261101)
        self.assertTrue(0 < len(dec) < len(inputs))

    def test_hidden_shape_13_tasks(self):
        dec, inputs = self.check(13, 1101)
        self.assertTrue(0 < len(dec) < len(inputs))

    def test_no_outputs_dirs_all_fallback(self):
        challenges = make_challenges(random.Random(3), 9, set())
        sub, _ = run_finalizer(challenges, None, None)
        self.assert_valid(challenges, sub)

    def test_passA_only_no_passB_dir(self):
        challenges = make_challenges(random.Random(4), 6, set())
        t = next(iter(challenges))
        sub, _ = run_finalizer(challenges, {f"{t}_0.permute0123456789": ([[[3, 4], [5, 6]]], [0.3])}, None)
        self.assert_valid(challenges, sub)
        self.assertEqual(sub[t][0]["attempt_1"], [[3, 4], [5, 6]])

    def test_passA_top1_restored_when_passB_outvotes_it(self):
        challenges = make_challenges(random.Random(6), 5, set())
        t = next(iter(challenges))
        a1, b1, b2 = [[1, 1]], [[2, 2]], [[3, 3]]
        plan_a = {f"{t}_0.permute0123456789": ([a1], [0.1])}
        plan_b = {f"{t}_0.p{j}": ([b1, b2], [0.2, 0.3]) for j in range(3)}  # B votes 3x for b1 and b2, A once for a1
        sub, out = run_finalizer(challenges, plan_a, plan_b)
        self.assert_valid(challenges, sub)
        self.assertEqual(sub[t][0]["attempt_1"], b1)
        self.assertEqual(sub[t][0]["attempt_2"], a1)  # pass-A top-1 forced into attempt_2
        self.assertIn("pass-A top1 restored=1", out)

    def test_default_never_promotes_and_flag_is_live(self):
        challenges = make_challenges(random.Random(7), 6, set())
        sub0, out0 = run_finalizer(challenges, None, None, expert_stub=True)
        self.assertTrue(all(e["attempt_2"] == [[0]] for v in sub0.values() for e in v))
        self.assertIn("promote=False", out0)
        sub1, out1 = run_finalizer(challenges, None, None, expert_stub=True, extra_env={"ARC_ASTRA_PROMOTE": "1"})
        self.assertTrue(all(e["attempt_2"] == [[7]] for v in sub1.values() for e in v))  # test is sensitive to the flag
        self.assertEqual([[e["attempt_1"] for e in v] for v in sub0.values()], [[e["attempt_1"] for e in v] for v in sub1.values()])

    def test_debug_ids_only_in_commit_branches(self):
        for cell in code_cells():
            for line in cell.splitlines():
                if any(d in line for d in DEBUG_IDS):
                    self.assertTrue("ARC_DEBUG_KEYS" in line, line)
        self.assertIn("list(arc_challenges) if RERUN_MODE", "".join(code_cells()))

    def test_rerun_budget_unchanged_commit_trimmed(self):
        code = "".join(code_cells())
        self.assertIn("MAX_SECONDS = 12 * 3600 - 900 if RERUN_MODE else 25 * 60", code)
        self.assertIn("min_b = 25 * 60 if RERUN_MODE else 10 * 60", code)


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-evidence"]:
        res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BoykoForkHiddenShape))
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(json.dumps({
            "passed": res.wasSuccessful(), "command": ".venv/bin/python tests/test_arc_agi_2_fork_boyko_hidden_shape.py --write-evidence",
            "tests_run": res.testsRun, "failures": len(res.failures), "errors": len(res.errors),
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
            "src_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SRC.glob("*.py"))}}, indent=1) + "\n")
        sys.exit(0 if res.wasSuccessful() else 1)
    unittest.main()
