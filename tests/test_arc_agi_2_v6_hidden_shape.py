"""v6 hidden-shape test: the real starter + assembly code on synthetic hidden test files.

A mock solver replaces only the GPU worker. Test files use fresh random IDs, 120/240/400
tasks, shuffled order and multiple test inputs; one worker crashes, one writes a truncated
result, one hangs past the deadline. The submission must cover every test input validly.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / os.environ.get("ARC_TEST_SRC", "runs/arc-agi-2/v6/src")  # v7+: ARC_TEST_SRC=runs/arc-agi-2/v7/src
PUBLIC_TEST = ROOT / "data/arc-agi-2/official-v1/arc-agi_test_challenges.json"
sys.path.insert(0, str(SRC))

MOCK_SOLVER = r'''
import bz2, hashlib, json, os, pickle, time
import numpy as np
import arc_v6_config as cfg

def answer(key, i):
    h = int(hashlib.sha256(f"{key}/{i}".encode()).hexdigest(), 16)
    return np.full((1 + h % 5, 1 + (h >> 8) % 7), (h >> 16) % 10, dtype=int)

def worker(rank, queue, end_time):
    out = os.environ["ARC_V6_OUTPUTS"]
    os.makedirs(out, exist_ok=True)
    with open(cfg.work_challenges_path()) as f:
        data = json.load(f)
    done = 0
    while not queue.empty():
        if time.time() > end_time:
            break
        key = queue.get()
        if key is None:
            break
        faults = os.environ.get("MOCK_FAULTS", "1") == "1"
        if faults and rank == 2 and done == 3:
            os._exit(1)                      # simulated CUDA crash: process dies
        if faults and rank == 1 and done == 5:
            time.sleep(3600)                 # hang past the deadline: parent must kill
        for i in range(len(data[key]["test"])):
            sub = f"{key}_{i}.transpose.permute0123456789"
            res = [{"beam_score": 0.1, "score_aug": [0.1] * 8, "solution": answer(key, i)},
                   {"beam_score": 0.9, "score_aug": [0.9] * 8, "solution": answer(key, i) * 0}]
            if faults and rank == 3 and done == 2:
                with open(os.path.join(out, sub), "wb") as f:
                    f.write(b"BZh91AY&SY truncated")
                continue
            tmp = os.path.join(out, "._tmp_" + sub)
            with bz2.BZ2File(tmp, "w") as f:
                pickle.dump(res, f)
            os.replace(tmp, os.path.join(out, sub))
        done += 1
        time.sleep(0.05)
'''


NOTEBOOK_RUNNER = r'''
import json, os, subprocess, sys
nb = json.load(open(sys.argv[1]))
ns = {}
sys.path.insert(0, os.getcwd())
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if src.startswith("%%writefile "):
        name, body = src[len("%%writefile "):].split("\n", 1)
        open(name.strip(), "w").write(body)
    elif src.startswith("!"):
        if src.startswith(("!pip", "!nvidia-smi")):
            continue
        cmd = src[1:].format(**ns)
        rc = subprocess.run(cmd, shell=True).returncode
        print("shell rc", rc, flush=True)
    else:
        exec(compile(src, "cell", "exec"), ns)
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


class HiddenShape(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.public = set(json.loads(PUBLIC_TEST.read_text())) if PUBLIC_TEST.exists() else set()
        (self.base / "mock").mkdir()
        (self.base / "mock/mock_solver.py").write_text(MOCK_SOLVER)

    def tearDown(self):
        self.tmp.cleanup()

    def mount(self, n, seed):
        rng = random.Random(seed)
        comp = self.base / f"input-{n}/competitions/arc-prize-2026-arc-agi-2"
        comp.mkdir(parents=True)
        test = synthetic(n, rng, self.public)
        evaluation = synthetic(12, rng, self.public | set(test))
        (comp / "arc-agi_test_challenges.json").write_text(json.dumps(test))
        (comp / "arc-agi_evaluation_challenges.json").write_text(json.dumps(evaluation))
        (comp / "arc-agi_evaluation_solutions.json").write_text(json.dumps(
            {k: [grid(rng) for _ in t["test"]] for k, t in evaluation.items()}))
        decoy = self.base / f"input-{n}/other-dataset"           # an unrelated mounted copy
        decoy.mkdir(parents=True)
        (decoy / "arc-agi_test_challenges.json").write_text("{}")
        model = self.base / f"input-{n}/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        return test, evaluation

    def run_pipeline(self, n, rerun, seconds=25):
        work = self.base / f"work-{n}-{rerun}"
        work.mkdir()
        env = dict(os.environ, ARC_V6_INPUT_ROOT=str(self.base / f"input-{n}"), ARC_V6_WORK_DIR=str(work),
                   ARC_V6_MARKER_DIR=str(work), ARC_V6_SOLVER_MODULE="mock_solver",
                   ARC_V6_OUTPUTS=str(work / "outputs"), ARC_V6_KILL_GRACE="3", ARC_V6_MARKER_POLL="0.2",
                   PYTHONPATH=os.pathsep.join([str(SRC), str(self.base / "mock")]))
        env.pop("KAGGLE_IS_COMPETITION_RERUN", None)
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
        return work, json.loads((work / "submission.json").read_text()), started.stdout

    def check_full_coverage(self, test, submission):
        import arc_v6_config as cfg
        self.assertEqual(set(submission), set(test))
        self.assertTrue(cfg.validate_submission(test, submission))
        for key, task in test.items():
            self.assertEqual(len(submission[key]), len(task["test"]))

    def test_rerun_hidden_shapes(self):
        sys.path.insert(0, str(self.base / "mock"))
        import mock_solver
        for n, seed in ((120, 1), (240, 2), (400, 3)):
            with self.subTest(n=n):
                test, _ = self.mount(n, seed)
                self.assertFalse(set(test) & self.public)
                work, submission, log = self.run_pipeline(n, rerun=True)
                self.check_full_coverage(test, submission)
                record = json.loads((work / "v6-assembly.json").read_text())
                self.assertIsNone(record["error"])
                self.assertEqual(record["test_tasks"], n)
                hits = sum(submission[k][i]["attempt_1"] == mock_solver.answer(k, i).tolist()
                           for k, t in test.items() for i in range(len(t["test"])))
                # Crashed/hung/truncated workers lose a few tasks; the others keep consuming the queue.
                self.assertGreater(hits, 0.5 * sum(len(t["test"]) for t in test.values()))
                self.assertIn('"tasks": %d' % n, log)
                exits = json.loads([l for l in log.splitlines() if "workers_exited" in l][0])["exitcodes"]
                self.assertEqual(exits[0], 0)
                self.assertNotEqual(exits[1], 0)   # hung worker was killed
                self.assertEqual(exits[2], 1)      # crashed worker
                total = sum(len(t["test"]) for t in test.values())
                self.assertLess(hits, total)       # the injected faults really happened

    def test_commit_smoke_writes_test_fallback(self):
        test, evaluation = self.mount(240, 7)
        work, submission, log = self.run_pipeline(240, rerun=False, seconds=15)
        self.check_full_coverage(test, submission)
        self.assertTrue(all(e["attempt_1"] == [[0]] for v in submission.values() for e in v))
        smoke = json.loads((work / "eval-smoke-submission.json").read_text())
        self.assertEqual(set(smoke), set(evaluation))
        self.assertIn('"tasks": 4', log)

    def test_built_notebook_cells_execute(self):
        """Execute the packaged notebook itself (magics emulated) on a hidden-shape mount."""
        uploads = sorted(SRC.parent.glob("upload-*/*.ipynb"))
        self.assertTrue(uploads, "build a notebook first")
        for nb_path in uploads:
            with self.subTest(notebook=nb_path.parent.name):
                n = 240
                test, _ = self.mount(n, 21) if not (self.base / f"input-{n}").exists() else (json.loads(
                    (self.base / f"input-{n}/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json").read_text()), None)
                work = self.base / f"nbwork-{nb_path.parent.name}"
                work.mkdir()
                runner = self.base / "run_notebook.py"
                runner.write_text(NOTEBOOK_RUNNER)
                env = dict(os.environ, ARC_V6_INPUT_ROOT=str(self.base / f"input-{n}"), ARC_V6_WORK_DIR=str(work),
                           ARC_V6_MARKER_DIR=str(work), ARC_V6_SOLVER_MODULE="mock_solver", MOCK_FAULTS="0",
                           ARC_V6_OUTPUTS=str(work / "outputs"), ARC_V6_MARKER_POLL="0.2",
                           KAGGLE_IS_COMPETITION_RERUN="True",
                           PATH=os.pathsep.join([str(Path(sys.executable).parent), os.environ["PATH"]]),
                           PYTHONPATH=os.pathsep.join([str(work), str(self.base / "mock")]))
                done = subprocess.run([sys.executable, str(runner), str(nb_path)], cwd=work, env=env,
                                      capture_output=True, text=True, timeout=300)
                self.assertEqual(done.returncode, 0, done.stdout[-3000:] + done.stderr[-3000:])
                submission = json.loads((work / "submission.json").read_text())
                self.check_full_coverage(test, submission)
                record = json.loads((work / "v6-assembly.json").read_text())
                self.assertIsNone(record["error"])
                self.assertEqual(record["decoded_files"], sum(len(t["test"]) for t in test.values()))

    def test_smoke_selection_uses_size_not_ids(self):
        import arc_v6_config as cfg
        rng = random.Random(11)
        tasks = synthetic(30, rng, set())
        chosen = cfg.select_keys(tasks, rerun=False)
        self.assertEqual(len(chosen), 4)
        largest = max(tasks, key=lambda k: (cfg._cells(tasks[k]), k))
        self.assertIn(largest, chosen)
        self.assertTrue(any(len(tasks[k]["test"]) > 1 for k in chosen))
        rerun_keys = cfg.select_keys(tasks, rerun=True)
        self.assertEqual(sorted(rerun_keys), sorted(tasks))          # every task exactly once, any order
        if hasattr(cfg, "adaptive_cap"):                             # v7+
            from unittest import mock
            with mock.patch.dict(os.environ, {"KAGGLE_IS_COMPETITION_RERUN": "1"}):
                self.assertEqual(cfg.adaptive_cap(43000, 0), cfg.TASK_SECONDS)
                self.assertEqual(cfg.adaptive_cap(100, 500), cfg.TASK_FLOOR_SECONDS)
                self.assertAlmostEqual(cfg.adaptive_cap(42000, 236), 700.0)
            with mock.patch.dict(os.environ, {"KAGGLE_IS_COMPETITION_RERUN": ""}):
                self.assertEqual(cfg.adaptive_cap(3000, 0), getattr(cfg, "COMMIT_TASK_SECONDS", cfg.TASK_SECONDS))
            self.assertEqual(cfg.estimated_work({"train": None, "test": [1]}), 0)   # odd shape never raises
            work = [cfg.estimated_work(tasks[k]) for k in rerun_keys]
            self.assertEqual(work, sorted(work))                     # cheapest first


if __name__ == "__main__":
    unittest.main()
