"""v9 pass-B test (v8 test adapted): the real starter + assembly on synthetic hidden test files with a mock GPU worker.

Pass A must run exactly as in v7; pass B may only use the wall time pass A leaves, must take tasks pass A left
without output first, then tasks pass A is unsure of (kgmon top-1 below the threshold), then sure ones, then
tasks that raised (unsure tasks a floor cap would cut before the others), truncated where predicted full-solve
seconds exceed the pass-B budget; any planning error falls back to v8's full queue. Pass B writes to its own store with its own seeds, and pooling keeps pass A's top-1. Run with:
  ARC_TEST_SRC=runs/arc-agi-2/v9/src ARC_V9_TEST_SRC=runs/arc-agi-2/v9/src .venv/bin/python -m unittest tests.test_arc_agi_2_v6_hidden_shape tests.test_arc_agi_2_v9_pass_b -v
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
SRC = ROOT / os.environ.get("ARC_V9_TEST_SRC", "runs/arc-agi-2/v9/src")
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

def a_sure(key):     # v9: pass A's top-1 has 8 votes (kgmon 7.9 >= 5) instead of 3 (kgmon 2.9)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 3 == 1

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
                res = sample(x, 8 if a_sure(key) else 3, 0.1) + sample((x + 1) % 10, 1, 0.5)
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
                work, sub, events = self.run_pipeline(root, ARC_V9_PASS_B_PLAN_SCALE="0.005")
                msgs = [e["msg"] for e in events]
                self.assertIn("pass_b_start", msgs)
                order_a = cfg.select_keys(test, rerun=True)
                missing = [k for k in order_a if m.a_misses(k) and not m.a_raises(k)]
                raised = [k for k in order_a if m.a_raises(k)]
                unsure = [k for k in order_a if not m.a_misses(k) and not m.a_sure(k)]
                long_ = [k for k in unsure if cfg.full_seconds(test[k]) > cfg.TASK_FLOOR_SECONDS]
                short = [k for k in unsure if k not in long_]
                sure = [k for k in order_a if not m.a_misses(k) and m.a_sure(k)]
                self.assertTrue(missing and raised and long_ and short and sure)
                start_b = next(e for e in events if e["msg"] == "pass_b_start")
                self.assertEqual(start_b["missing_after_a"], len(missing) + len(raised))
                self.assertEqual(start_b["failed_after_a"], len(raised))
                self.assertEqual(start_b["pass_b_order"], "confidence")
                self.assertEqual((start_b["missing"], start_b["unsure_long"], start_b["unsure_short"], start_b["sure"],
                                  start_b["raised"]), (len(missing), len(long_), len(short), len(sure), len(raised)))
                limit = start_b["limit"]
                self.assertTrue(len(missing) < limit < n - len(raised), limit)   # truncated below the full queue
                planned = (missing + long_ + short + sure + raised)[:limit]
                self.assertEqual(start_b["tasks"], limit)
                self.assertLessEqual(start_b["planned_sum"], start_b["budget"] + 1)
                # the same plan offline, in production scale, with ample and with no time
                plan, info = cfg.pass_b_plan(order_a, str(work / "outputs"), test, 1e7)
                self.assertEqual((info["pass_b_order"], plan), ("confidence", missing + long_ + short + sure + raised))
                plan, info = cfg.pass_b_plan(order_a, str(work / "outputs"), test, 0.0)
                self.assertEqual(plan, (missing + long_ + short + sure + raised)[:cfg.NUM_WORKERS])
                exits = [e for e in events if e["msg"] == "workers_exited"]
                self.assertEqual([e["pass_label"] for e in exits], ["A", "B"])
                # Pass B runs exactly the planned tasks: pass-A-empty first, then unsure (cheap-first within groups).
                order_b = [l.split()[1] for l in sorted((work / "mocklog/order-B.txt").read_text().splitlines())]
                self.assertEqual(sorted(order_b), sorted(planned))
                # (workers log after dequeuing, so allow one in-flight task per worker of slack)
                self.assertLess(max(order_b.index(k) for k in missing), len(missing) + cfg.NUM_WORKERS)
                self.assertLess(max(order_b.index(k) for k in long_ if k in planned),
                                min(len(missing) + len(long_), limit) + cfg.NUM_WORKERS)
                self.assertFalse(set(raised) & set(order_b))
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
                in_b = set(planned)
                self.assertEqual(record["changed_vs_pass_a"], sum(len(test[k]["test"]) for k in in_b))
                for k, t in test.items():
                    for i in range(len(t["test"])):
                        x = m.answer(k, i)
                        got = [sub[k][i]["attempt_1"], sub[k][i]["attempt_2"]]
                        if m.a_misses(k):
                            # no pass-A candidate: pass B's own top-2, else the fallback grid
                            self.assertEqual(got, [((x + 2) % 10).tolist(), x.tolist()] if k in in_b else [[[0]], [[0]]])
                        elif k in in_b:
                            # pass-A top-1 kept as attempt_1; pass B's strong grid replaces pass-A #2
                            self.assertEqual(got, [x.tolist(), ((x + 2) % 10).tolist()])
                        else:
                            self.assertEqual(got, [x.tolist(), ((x + 1) % 10).tolist()])   # not redrawn: v7 choice

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
        work, sub, events = self.run_pipeline(root, seconds=40, MOCK_B_HANG="1", ARC_V8_PASS_B_DEADLINE_MARGIN="15",
                                              ARC_V9_PASS_B_PLAN_SCALE="0.001")
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
        start_b = next(e for e in events if e["msg"] == "pass_b_start")
        import arc_v6_config as cfg
        # v9: the smoke's few tasks all reach pass B (limit >= NUM_WORKERS), so the audit still sees 4 B ranks.
        self.assertEqual(start_b["pass_b_order"], "confidence")
        self.assertEqual(start_b["tasks"], cfg.SMOKE_TASKS)
        order_b = (work / "mocklog/order-B.txt").read_text().split()[1::2]
        self.assertEqual(len(order_b), cfg.SMOKE_TASKS)
        record = json.loads((work / "v6-assembly.json").read_text())
        self.assertIsNone(record["error"])

    def store(self, files):
        import bz2, pickle
        out = self.base / "store"
        out.mkdir(exist_ok=True)
        for name, samples in files.items():
            with bz2.BZ2File(out / name, "w") as f:
                pickle.dump(samples, f)
        return out

    def test_plan_confidence_groups_and_fallbacks(self):
        import arc_v6_config as cfg
        g = lambda v, shape=(2, 2): np.full(shape, v)
        s = lambda grid, n, aug=0.1: [{"beam_score": 0.1, "score_aug": [aug] * 8, "solution": grid} for _ in range(n)]
        sq = lambda side: [[1] * side for _ in range(side)]
        task = lambda side, tests=1: {"train": [{"input": sq(side), "output": sq(side)} for _ in range(3)],
                                      "test": [{"input": sq(side)} for _ in range(tests)]}
        challenges = {"sure": task(20), "half": task(20, 2), "multi": task(20, 2), "low": task(1), "gone": task(20),
                      "boom": task(20), "nan": task(1), "junk": task(20)}
        self.assertLess(cfg.full_seconds(task(1)), cfg.TASK_FLOOR_SECONDS)      # a floor cap never cuts it
        self.assertGreater(cfg.full_seconds(task(20)), cfg.TASK_FLOOR_SECONDS)  # a floor cap cuts it (v9 and v11)
        self.assertEqual(cfg.full_seconds({"test": []}), cfg.TASK_SECONDS)      # odd shape: author cap
        out = self.store({
            "sure_0.rot90.permute0123456789": s(g(1), 4),
            "sure_0.transpose.permute0123456789": s(g(1), 4) + s(g(2), 1),     # 8 votes across files: 7.9
            "half_0.permute0123456789": s(g(3), 9),                              # half_1 has no file: -inf
            "multi_0.permute0123456789": s(g(4), 9),
            "multi_1.permute0123456789": s(g(5), 9, aug=0.2),
            "low_0.permute0123456789": s(g(6), 2),
            "nan_0.permute0123456789": [{"beam_score": 0.1, "score_aug": [], "solution": g(7)}],
            "._tmp_low_0.x": s(g(6), 50),                                        # unfinished write: ignored
        })
        (out / "junk_0.permute0123456789").write_bytes(b"BZh91AY&SY truncated")  # unreadable: covered, -inf
        (out.parent / "store_failed.txt").write_text(json.dumps("boom") + "\n")
        keys = ["low", "sure", "half", "gone", "multi", "boom", "nan", "junk"]
        conf = cfg.task_confidence(keys, str(out), challenges, time.time() + 60)
        self.assertAlmostEqual(conf["sure"], 7.9)
        self.assertAlmostEqual(conf["multi"], 8.8)
        self.assertAlmostEqual(conf["low"], 1.9)
        self.assertEqual(conf["half"], float("-inf"))
        self.assertEqual(conf["gone"], float("-inf"))
        self.assertEqual(conf["junk"], float("-inf"))
        self.assertTrue(np.isnan(conf["nan"]) or conf["nan"] == float("-inf"))
        # missing (gone) -> unsure a floor cap cuts (half, junk) -> other unsure (low, nan) -> sure -> raised
        order = ["gone", "half", "junk", "low", "nan", "sure", "multi", "boom"]
        plan, info = cfg.pass_b_plan(keys, str(out), challenges, 1e7)
        self.assertEqual((info["pass_b_order"], plan), ("confidence", order))
        self.assertEqual((info["missing"], info["unsure_long"], info["unsure_short"], info["sure"], info["raised"]),
                         (1, 2, 2, 2, 1))
        plan, info = cfg.pass_b_plan(keys, str(out), challenges, 0.0)        # no budget: one task per worker
        self.assertEqual((plan, info["limit"]), (order[:cfg.NUM_WORKERS], cfg.NUM_WORKERS))
        need = [min(cfg.TASK_SECONDS, max(cfg.TASK_FLOOR_SECONDS, cfg.full_seconds(challenges[k]))) for k in order]
        budget = sum(need[:6]) + 0.5 * need[6]                               # room for exactly six
        plan, info = cfg.pass_b_plan(keys, str(out), challenges, budget / cfg.NUM_WORKERS + cfg.PASS_B_START_STAGGER)
        self.assertEqual(plan, order[:6])
        self.assertLessEqual(info["planned_sum"], info["budget"] + 1)
        # a key the challenge file lacks, a missing store, or a slow read: v8's full queue
        for args, why in (((keys + ["ghost"], str(out), challenges), "KeyError"),
                          ((keys, str(out) + "-absent", challenges), "FileNotFoundError")):
            plan, info = cfg.pass_b_plan(*args, 1.0)
            self.assertEqual(info["pass_b_order"], "fallback")
            self.assertIn(why, info["error"])
            self.assertEqual(plan, cfg.pass_b_keys(args[0], args[1]))
        old = cfg.PASS_B_PLAN_SECONDS
        try:
            cfg.PASS_B_PLAN_SECONDS = -1
            plan, info = cfg.pass_b_plan(keys, str(out), challenges, 1.0)
        finally:
            cfg.PASS_B_PLAN_SECONDS = old
        self.assertEqual(info["pass_b_order"], "fallback")
        self.assertIn("TimeoutError", info["error"])
        self.assertEqual(plan, cfg.pass_b_keys(keys, str(out)))

    def test_plan_on_real_colab_store(self):
        """The real pass-A store from the Colab calibration (10 public-eval tasks): parsed, and the task a second
        full-cap seed solved there (800d221b) is planned among the unsure tasks a floor cap would cut."""
        import tarfile
        import arc_v6_config as cfg
        tgz = ROOT / "artifacts/verification/arc-agi-2/colab-cal-v1/out_a_run2.tgz"
        ev_path = ROOT / "data/arc-agi-2/official-v1/arc-agi_evaluation_challenges.json"
        if not (tgz.exists() and ev_path.exists()):
            self.skipTest("local Colab store or public evaluation file absent")
        with tarfile.open(tgz) as tar:
            tar.extractall(self.base, filter="data")
        ev = json.loads(ev_path.read_text())
        keys = [k for k in cfg.select_keys(ev, rerun=True) if k in json.loads(
            (ROOT / "runs/arc-agi-2/colab-pair-v1/keys_b.json").read_text())]
        challenges = {k: ev[k] for k in keys}
        plan, info = cfg.pass_b_plan(keys, str(self.base / "out_a"), challenges, 4500.0)
        self.assertEqual(info["pass_b_order"], "confidence")
        self.assertEqual(info["missing"] + info["unsure_long"] + info["unsure_short"] + info["sure"], len(keys))
        self.assertIn("800d221b", plan)
        if cfg.full_seconds(ev["800d221b"]) > cfg.TASK_FLOOR_SECONDS:   # v9 (full cost); v11 half cost predicts < floor
            self.assertIn("800d221b", plan[:info["missing"] + info["unsure_long"]])
        self.assertLessEqual(info["planned_sum"], info["budget"] + 1)
        self.assertLess(info["plan_seconds"], 10)

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
