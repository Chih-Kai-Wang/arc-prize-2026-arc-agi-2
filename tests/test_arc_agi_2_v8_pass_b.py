"""v8 pass-B test: the real starter + assembly on synthetic hidden test files with a mock GPU worker.

Pass A must run exactly as in v7; pass B may only use the wall time pass A leaves, must take tasks pass A
left without output first, must write to its own store with its own seeds, and pooling must keep pass A's
top-1 while letting pass B fill the other attempt. Run with the v6 hidden-shape test too:
  ARC_TEST_SRC=runs/arc-agi-2/v8/src .venv/bin/python -m unittest tests.test_arc_agi_2_v6_hidden_shape tests.test_arc_agi_2_v8_pass_b -v
"""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / os.environ.get("ARC_V8_TEST_SRC", "runs/arc-agi-2/v8/src")
PUBLIC_TEST = ROOT / "data/arc-agi-2/official-v1/arc-agi_test_challenges.json"

MOCK_SOLVER = r'''
import bz2, hashlib, json, os, pickle, time
import numpy as np
import arc_v6_config as cfg

def answer(key, i):
    h = int(hashlib.sha256(f"{key}/{i}".encode()).hexdigest(), 16)
    return np.full((1 + h % 5, 1 + (h >> 8) % 7), (h >> 16) % 10, dtype=int)

def a_misses(key):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 7 == 0

def a_raises(key):   # subset of a_misses: the solve raised (solver calls cfg.mark_failed)
    return a_misses(key) and int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0

def sample(grid, n, aug):
    return [{"beam_score": 0.1, "score_aug": [aug] * 8, "solution": grid} for _ in range(n)]

def worker(rank, queue, end_time):
    out = cfg.outputs_dir()
    os.makedirs(out, exist_ok=True)
    pass_b = os.environ.get("ARC_V8_TRAIN_AUG_SEED") is not None
    with open(cfg.work_challenges_path()) as f:
        data = json.load(f)
    seen = [k for k in ("ARC_V8_TRAIN_AUG_SEED", "ARC_V8_EVAL_AUG_SEED", "ARC_V8_TRAINER_SEED",
                        "ARC_V8_LORA_SEED", "ARC_V8_SCORE_SEED_OFFSET")]
    with open(os.path.join(os.environ["MOCK_LOG"], f"env-{'B' if pass_b else 'A'}-{rank}.json"), "w") as f:
        json.dump({"out": out, "seeds": {k: os.environ.get(k) for k in seen}}, f)
    done = 0
    while not queue.empty():
        if time.time() > end_time:
            break
        key = queue.get()
        if key is None:
            break
        with open(os.path.join(os.environ["MOCK_LOG"], f"order-{'B' if pass_b else 'A'}.txt"), "a") as f:
            f.write(f"{time.time():.6f} {key}\n")
        if pass_b and rank == 1 and done == 2 and os.environ.get("MOCK_B_HANG") == "1":
            time.sleep(3600)                 # pass-B hang past the deadline: parent must kill
        if not pass_b and a_misses(key):
            if a_raises(key):
                cfg.mark_failed(key)
            done += 1
            continue                         # pass A leaves this task without any output
        for i in range(len(data[key]["test"])):
            x = answer(key, i)
            if pass_b:
                res = sample((x + 2) % 10, 6, 0.1) + sample(x, 1, 0.2)
                if os.environ.get("MOCK_B_CORRUPT") == "1":
                    res = [{"beam_score": 0.1}]          # malformed sample: pooling must fall back
            else:
                res = sample(x, 3, 0.1) + sample((x + 1) % 10, 1, 0.5)
            sub = f"{key}_{i}.transpose.permute0123456789"
            tmp = os.path.join(out, "._tmp_" + sub)
            with bz2.BZ2File(tmp, "w") as f:
                pickle.dump(res, f)
            os.replace(tmp, os.path.join(out, sub))
        done += 1
        time.sleep(0.02)
'''


def grid(rng):
    h, w = rng.randint(1, 30), rng.randint(1, 30)
    return [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]


def synthetic(n, rng, banned):
    tasks = {}
    while len(tasks) < n:
        key = "%08x" % rng.getrandbits(32)
        if key in banned or key in tasks:
            continue
        tests = rng.choice([1, 1, 1, 2, 3])
        tasks[key] = {"train": [{"input": grid(rng), "output": grid(rng)} for _ in range(rng.randint(2, 4))],
                      "test": [{"input": grid(rng)} for _ in range(tests)]}
    items = list(tasks.items())
    rng.shuffle(items)
    return dict(items)


class PassB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.public = set(json.loads(PUBLIC_TEST.read_text())) if PUBLIC_TEST.exists() else set()
        (self.base / "mock").mkdir()
        (self.base / "mock/mock_solver_b.py").write_text(MOCK_SOLVER)
        sys.path[:0] = [str(SRC), str(self.base / "mock")]

    def tearDown(self):
        for p in (str(SRC), str(self.base / "mock")):
            if p in sys.path:
                sys.path.remove(p)
        self.tmp.cleanup()

    def mount(self, n, seed):
        rng = random.Random(seed)
        comp = self.base / f"input-{n}-{seed}/competitions/arc-prize-2026-arc-agi-2"
        comp.mkdir(parents=True)
        test = synthetic(n, rng, self.public)
        evaluation = synthetic(12, rng, self.public | set(test))
        (comp / "arc-agi_test_challenges.json").write_text(json.dumps(test))
        (comp / "arc-agi_evaluation_challenges.json").write_text(json.dumps(evaluation))
        (comp / "arc-agi_evaluation_solutions.json").write_text(json.dumps(
            {k: [grid(rng) for _ in t["test"]] for k, t in evaluation.items()}))
        model = self.base / f"input-{n}-{seed}/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        return test, self.base / f"input-{n}-{seed}"

    def run_pipeline(self, root, rerun=True, seconds=40, **extra):
        work = Path(tempfile.mkdtemp(dir=self.base))
        logs = work / "mocklog"
        logs.mkdir()
        env = dict(os.environ, ARC_V6_INPUT_ROOT=str(root), ARC_V6_WORK_DIR=str(work), ARC_V6_MARKER_DIR=str(work),
                   ARC_V6_SOLVER_MODULE="mock_solver_b", ARC_V6_OUTPUTS=str(work / "outputs"),
                   ARC_V6_KILL_GRACE="3", ARC_V6_MARKER_POLL="0.2",
                   MOCK_LOG=str(logs), PYTHONPATH=os.pathsep.join([str(SRC), str(self.base / "mock")]),
                   **{"ARC_V8_PASS_B_MIN_SECONDS": "5", "ARC_V8_PASS_B_DEADLINE_MARGIN": "0", **extra})
        for k in ("KAGGLE_IS_COMPETITION_RERUN", "ARC_V8_OUTPUTS_B", "ARC_V8_TRAIN_AUG_SEED", "ARC_V8_EVAL_AUG_SEED",
                  "ARC_V8_TRAINER_SEED", "ARC_V8_LORA_SEED", "ARC_V8_SCORE_SEED_OFFSET"):
            if k not in extra:
                env.pop(k, None)
        if rerun:
            env["KAGGLE_IS_COMPETITION_RERUN"] = "True"
        start = time.time()
        started = subprocess.run([sys.executable, str(SRC / "starter.py"), "--end-time", str(start + seconds)],
                                 cwd=work, env=env, capture_output=True, text=True, timeout=seconds + 60)
        self.assertEqual(started.returncode, 0, started.stderr[-2000:])
        self.assertLess(time.time() - start, seconds + 20, "parent must kill hung workers")
        assembled = subprocess.run([sys.executable, "-c", "import assemble; assemble.main()"], cwd=work, env=env,
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(assembled.returncode, 0, assembled.stderr[-2000:])
        events = [json.loads(l) for l in started.stdout.splitlines() if l.startswith("{")]
        return work, json.loads((work / "submission.json").read_text()), events

    def test_pass_b_pools_and_keeps_pass_a_top1(self):
        import mock_solver_b as m
        import arc_v6_config as cfg
        for n, seed in ((120, 31), (240, 32)):
            with self.subTest(n=n):
                test, root = self.mount(n, seed)
                self.assertFalse(set(test) & self.public)
                work, sub, events = self.run_pipeline(root)
                msgs = [e["msg"] for e in events]
                self.assertIn("pass_b_start", msgs)
                order_a = cfg.select_keys(test, rerun=True)
                missing = [k for k in order_a if m.a_misses(k) and not m.a_raises(k)]
                raised = [k for k in order_a if m.a_raises(k)]
                self.assertTrue(missing and raised)
                start_b = next(e for e in events if e["msg"] == "pass_b_start")
                self.assertEqual(start_b["missing_after_a"], len(missing) + len(raised))
                self.assertEqual(start_b["failed_after_a"], len(raised))
                self.assertEqual(cfg.pass_b_keys(order_a, str(work / "outputs")),
                                 missing + [k for k in order_a if not m.a_misses(k)] + raised)
                self.assertEqual(start_b["tasks"], n)
                exits = [e for e in events if e["msg"] == "workers_exited"]
                self.assertEqual([e["pass_label"] for e in exits], ["A", "B"])
                # Pass B takes tasks pass A left empty first (cheap-first within each group).
                order_b = [l.split()[1] for l in sorted((work / "mocklog/order-B.txt").read_text().splitlines())]
                # (workers log after dequeuing, so allow one in-flight task per worker of slack)
                self.assertLess(max(order_b.index(k) for k in missing), len(missing) + cfg.NUM_WORKERS)
                self.assertGreaterEqual(min(order_b.index(k) for k in raised), n - len(raised) - cfg.NUM_WORKERS)
                # Pass B used its own seeds and store; pass A the v7 defaults and the v7 store.
                for rank in range(cfg.NUM_WORKERS):
                    env_a = json.loads((work / f"mocklog/env-A-{rank}.json").read_text())
                    env_b = json.loads((work / f"mocklog/env-B-{rank}.json").read_text())
                    self.assertEqual(env_a["out"], str(work / "outputs"))
                    self.assertTrue(all(v is None for v in env_a["seeds"].values()))
                    self.assertEqual(env_b["out"], str(work / "outputs_b"))
                    self.assertEqual(env_b["seeds"], cfg.PASS_B_ENV)
                self.assertFalse(any(".b" in p.name for p in (work / "outputs").iterdir()))
                record = json.loads((work / "v6-assembly.json").read_text())
                self.assertIsNone(record["error"])
                self.assertGreater(record["pass_b_files"], 0)
                self.assertNotIn("pooling_error", record)
                total = sum(len(t["test"]) for t in test.values())
                self.assertEqual(record["changed_vs_pass_a"], total)
                for k, t in test.items():
                    for i in range(len(t["test"])):
                        x = m.answer(k, i)
                        got = [sub[k][i]["attempt_1"], sub[k][i]["attempt_2"]]
                        if m.a_misses(k):
                            # no pass-A candidate: pass B's own top-2
                            self.assertEqual(got, [((x + 2) % 10).tolist(), x.tolist()])
                        else:
                            # pass-A top-1 kept as attempt_1; pass B's strong grid replaces pass-A #2
                            self.assertEqual(got, [x.tolist(), ((x + 2) % 10).tolist()])

    def test_saturated_run_is_v7(self):
        """No time left after pass A: pass B never starts and the submission is pass A's top-2."""
        import mock_solver_b as m
        test, root = self.mount(120, 33)
        work, sub, events = self.run_pipeline(root, seconds=20, ARC_V8_PASS_B_MIN_SECONDS="100000")
        msgs = [e["msg"] for e in events]
        self.assertIn("pass_b_skipped", msgs)
        self.assertNotIn("pass_b_start", msgs)
        self.assertFalse((work / "outputs_b").exists())
        record = json.loads((work / "v6-assembly.json").read_text())
        self.assertEqual(record["changed_vs_pass_a"], 0)
        for k, t in test.items():
            for i in range(len(t["test"])):
                x = m.answer(k, i)
                want = [[[0]], [[0]]] if m.a_misses(k) else [x.tolist(), ((x + 1) % 10).tolist()]
                self.assertEqual([sub[k][i]["attempt_1"], sub[k][i]["attempt_2"]], want)

    def test_hung_pass_b_worker_is_killed_at_deadline(self):
        test, root = self.mount(240, 34)
        start = time.time()
        work, sub, events = self.run_pipeline(root, seconds=40, MOCK_B_HANG="1", ARC_V8_PASS_B_DEADLINE_MARGIN="15")
        # pass B runs to end_time - margin (+ 3 s kill grace), leaving the margin for assembly
        b_exit = next(e for e in events if e["msg"] == "workers_exited" and e["pass_label"] == "B")
        self.assertLess(b_exit["t"] - start, 40 - 15 + 3 + 6)
        exits = [e for e in events if e["msg"] == "workers_exited"]
        self.assertEqual([e["pass_label"] for e in exits], ["A", "B"])
        self.assertNotEqual(exits[1]["exitcodes"][1], 0)
        import arc_v6_config as cfg
        self.assertTrue(cfg.validate_submission(test, sub))

    def test_pooling_failure_falls_back_to_pass_a(self):
        import mock_solver_b as m
        test, root = self.mount(120, 35)
        work, sub, events = self.run_pipeline(root, MOCK_B_CORRUPT="1")
        record = json.loads((work / "v6-assembly.json").read_text())
        self.assertIsNone(record["error"])
        self.assertIn("pooling_error", record)
        for k, t in test.items():
            if m.a_misses(k):
                continue
            for i in range(len(t["test"])):
                x = m.answer(k, i)
                self.assertEqual([sub[k][i]["attempt_1"], sub[k][i]["attempt_2"]], [x.tolist(), ((x + 1) % 10).tolist()])

    def test_commit_smoke_runs_pass_b_and_reports_both(self):
        test, root = self.mount(240, 36)
        work, sub, events = self.run_pipeline(root, rerun=False, seconds=30)
        self.assertTrue(all(e["attempt_1"] == [[0]] for v in sub.values() for e in v))
        msgs = [e["msg"] for e in events]
        self.assertIn("pass_b_start", msgs)
        record = json.loads((work / "v6-assembly.json").read_text())
        self.assertIsNone(record["error"])

    def test_keep_a_top1_rule(self):
        import assemble
        X, Y, Z, W = (np.full((2, 2), v) for v in (1, 2, 3, 4))
        eq = lambda got, want: self.assertTrue(len(got) == len(want) and all(np.array_equal(a, b) for a, b in zip(got, want)))
        sel, changed = assemble.keep_a_top1({"t_0": [X, Y]}, {"t_0": [X, Y]})           # no pass B: v7 choice
        eq(sel["t_0"], [X, Y]); self.assertEqual(changed, 0)
        sel, _ = assemble.keep_a_top1({"t_0": [X, Y]}, {"t_0": [Z, X, Y]})              # B promotes Z
        eq(sel["t_0"], [X, Z])
        sel, _ = assemble.keep_a_top1({"t_0": [X, Y]}, {"t_0": [Z, W, X, Y]})           # A top-1 demoted: kept
        eq(sel["t_0"], [X, Z])
        sel, _ = assemble.keep_a_top1({}, {"t_0": [Z, W]})                               # only B ran
        eq(sel["t_0"], [Z, W])
        sel, _ = assemble.keep_a_top1({"t_0": [X]}, {"t_0": [X]})                        # singleton stays single
        eq(sel["t_0"], [X])
        sel, _ = assemble.keep_a_top1({"t_0": [X]}, {"t_0": [X, Z]})                     # B fills attempt_2
        eq(sel["t_0"], [X, Z])


if __name__ == "__main__":
    unittest.main()
