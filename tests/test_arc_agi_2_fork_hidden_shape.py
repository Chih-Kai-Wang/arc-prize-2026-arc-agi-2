"""Hidden-data-shape test for the AS-IS fork of mikelou1/arc-agi2-lb33-89-minimal-perfpatch.

No GPU, no model. The fork's real extracted loader/decoder modules and the notebook's real final assembly cell
(executed verbatim, only the /kaggle paths redirected to a temp dir) run on synthetic test files with fresh random
IDs, 7 and 13 tasks, shuffled order, 1 and 2 test inputs per task and varied grid sizes. Fake decoded-result files
exist for some test inputs and not for others (fallback path). The real starter.py queue builder runs under torch stubs.
Run:  .venv/bin/python -m unittest tests.test_arc_agi_2_fork_hidden_shape -v
Evidence JSON:  .venv/bin/python tests/test_arc_agi_2_fork_hidden_shape.py --write-evidence
"""
import bz2
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "runs/arc-agi-2/fork-perfpatch"
SRC = FORK / "src"
NOTEBOOK = FORK / "upload-arc-2026-fork-perfpatch-v1/arc-2026-fork-perfpatch-v1.ipynb"
COMP_DIR = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
OUT_DIR = "/kaggle/inference_outputs"
PUBLIC_TEST = ROOT / "data/arc-agi-2/official-v1/arc-agi_test_challenges.json"
PUBLIC_EVAL = ROOT / "data/arc-agi-2/official-v1/arc-agi_evaluation_challenges.json"
HARDCODED_SMOKE_IDS = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]

try:  # the loader only needs AutoTokenizer for a type annotation; stub it when transformers is absent
    import transformers  # noqa: F401
except ImportError:
    stub = types.ModuleType("transformers")
    stub.AutoTokenizer = type("AutoTokenizer", (), {})
    sys.modules["transformers"] = stub
sys.path.insert(0, str(SRC))
from arc_loader import ArcDataset  # noqa: E402
from arc_decoder import ArcDecoder  # noqa: E402


def final_cell_source():
    notebook = json.loads(NOTEBOOK.read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    cell = cells[-1]
    assert "decoder.run_selection_algo()" in cell and "submission.json" in cell
    return cell


def rand_grid(rng, rows, cols):
    return [[rng.randrange(10) for _ in range(cols)] for _ in range(rows)]


def make_challenges(rng, n_tasks, taken):
    tasks = {}
    while len(tasks) < n_tasks:
        tid = "".join(rng.choice("0123456789abcdef") for _ in range(8))
        if tid in taken or tid in tasks:
            continue
        r, c = rng.randint(1, 30), rng.randint(1, 30)
        tasks[tid] = {"train": [{"input": rand_grid(rng, r, c), "output": rand_grid(rng, rng.randint(1, 30), rng.randint(1, 30))}
                                for _ in range(rng.randint(2, 4))],
                      "test": [{"input": rand_grid(rng, rng.randint(1, 30), rng.randint(1, 30))}
                               for _ in range(rng.choice([1, 2]))]}
    items = list(tasks.items())
    rng.shuffle(items)
    return dict(items)


def write_decoded(directory, key, solutions, aug_means):
    """One decoder pickle in the exact format ArcDecoder.load_decoded_results expects (list of sample dicts)."""
    rows = [{"beam_score": 0.1 * (i + 1), "score_aug": [m] * 8, "solution": np.array(s, dtype=int)}
            for i, (s, m) in enumerate(zip(solutions, aug_means))]
    with bz2.BZ2File(os.path.join(directory, key), "w") as f:
        pickle.dump(rows, f)


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


@contextmanager
def workdir():
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(old)


def run_final_cell(challenges, decoded_plan, rerun=True, decoded_dir_exists=True):
    """Execute the notebook's real last cell with /kaggle paths redirected. Returns parsed submission.json."""
    with workdir() as tmp:
        comp = tmp / "comp"
        out = tmp / "inference_outputs"
        comp.mkdir()
        (comp / "arc-agi_test_challenges.json").write_text(json.dumps(challenges))
        if decoded_dir_exists:
            out.mkdir()
            for key, (solutions, means) in decoded_plan.items():
                write_decoded(out, key, solutions, means)
        source = final_cell_source().replace(COMP_DIR, str(comp)).replace(OUT_DIR, str(out))
        env = {"KAGGLE_IS_COMPETITION_RERUN": "1"} if rerun else {}
        with mock.patch.dict(os.environ, env, clear=False):
            if not rerun:
                os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
            exec(compile(source, "final_cell", "exec"), {"__name__": "__main__"})
        return json.loads((tmp / "submission.json").read_text())


def valid_grid(g):
    return (isinstance(g, list) and len(g) > 0 and all(isinstance(r, list) and len(r) > 0 for r in g)
            and len({len(r) for r in g}) == 1 and all(type(x) is int and 0 <= x <= 9 for r in g for x in r))


def assert_valid_submission(case, challenges, submission):
    case.assertEqual(set(submission), set(challenges), "missing or extra task ids")
    for tid, task in challenges.items():
        case.assertEqual(len(submission[tid]), len(task["test"]), tid)
        for entry in submission[tid]:
            case.assertEqual(set(entry), {"attempt_1", "attempt_2"})
            for attempt in entry.values():
                case.assertTrue(valid_grid(attempt), (tid, attempt))


class ForkHiddenShape(unittest.TestCase):
    def check_shape(self, n_tasks, seed):
        rng = random.Random(seed)
        # public release: the competition files are optional (same fallback as test_arc_agi_2_v6_hidden_shape.py)
        public = set(HARDCODED_SMOKE_IDS)
        for f in (PUBLIC_TEST, PUBLIC_EVAL):
            if f.exists():
                public |= set(json.loads(f.read_text()))
        challenges = make_challenges(rng, n_tasks, public)
        self.assertFalse(set(challenges) & public)
        plan, expected = {}, {}
        inputs = [(tid, i) for tid, t in challenges.items() for i in range(len(t["test"]))]
        decoded = set(rng.sample(inputs, len(inputs) // 2))
        for tid, i in inputs:
            if (tid, i) not in decoded:
                continue
            best, second, third = (rand_grid(rng, rng.randint(1, 30), rng.randint(1, 30)) for _ in range(3))
            # same base key from several subkeys (different augmentation suffixes), as the real worker writes them
            plan[f"{tid}_{i}.permute0123456789"] = ([second, best], [0.9, 0.2])
            plan[f"{tid}_{i}.rot90.permute9876543210"] = ([third], [1.5])
            expected[(tid, i)] = (best, second)
        # partially decoded task: only input 0 decoded, the others fall back (covered by random choice above)
        submission = run_final_cell(challenges, plan)
        assert_valid_submission(self, challenges, submission)
        for tid, i in inputs:
            entry = submission[tid][i]
            if (tid, i) in expected:
                best, second = expected[(tid, i)]
                self.assertEqual(entry["attempt_1"], best)
                self.assertEqual(entry["attempt_2"], second)
            else:
                self.assertEqual(entry, {"attempt_1": [[0]], "attempt_2": [[0]]})
        return challenges, submission, decoded, inputs

    def test_hidden_shape_7_tasks(self):
        _, _, decoded, inputs = self.check_shape(7, 20261013)
        self.assertTrue(0 < len(decoded) < len(inputs))

    def test_hidden_shape_13_tasks(self):
        _, _, decoded, inputs = self.check_shape(13, 77)
        self.assertTrue(0 < len(decoded) < len(inputs))

    def test_both_input_counts_present(self):
        challenges = make_challenges(random.Random(5), 13, set())
        counts = {len(t["test"]) for t in challenges.values()}
        self.assertEqual(counts, {1, 2})

    def test_no_decoded_files_all_fallback(self):
        challenges = make_challenges(random.Random(11), 9, set())
        submission = run_final_cell(challenges, {})
        assert_valid_submission(self, challenges, submission)
        self.assertTrue(all(e == {"attempt_1": [[0]], "attempt_2": [[0]]} for v in submission.values() for e in v))

    def test_single_candidate_leaves_attempt_2_fallback(self):
        challenges = make_challenges(random.Random(12), 5, set())
        tid = next(iter(challenges))
        submission = run_final_cell(challenges, {f"{tid}_0.permute0123456789": ([[[3, 4], [5, 6]]], [0.3])})
        assert_valid_submission(self, challenges, submission)
        self.assertEqual(submission[tid][0], {"attempt_1": [[3, 4], [5, 6]], "attempt_2": [[0]]})

    def test_rerun_missing_outputs_dir_raises(self):
        """Known risk, documented: no /kaggle/inference_outputs at all (every worker died before makedirs) fails the cell."""
        challenges = make_challenges(random.Random(13), 5, set())
        with self.assertRaises(FileNotFoundError):
            run_final_cell(challenges, {}, decoded_dir_exists=False)

    def test_decoder_api_matches_assembly_cell(self):
        challenges = make_challenges(random.Random(14), 6, set())
        with workdir() as tmp:
            path = tmp / "c.json"
            path.write_text(json.dumps(challenges))
            data = ArcDataset.from_file(str(path))
            decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
            (tmp / "o").mkdir()
            decoder.load_decoded_results(str(tmp / "o"))
            self.assertEqual(decoder.decoded_results, {})
            submission = data.get_submission(decoder.run_selection_algo())
        assert_valid_submission(self, challenges, submission)

    def test_rerun_queue_is_dynamic_and_complete(self):
        """starter.py in rerun mode queues every key of whatever test file it is given (sorted) plus 4 stop markers."""
        for n, seed in ((7, 1), (13, 2)):
            challenges = make_challenges(random.Random(seed), n, set(HARDCODED_SMOKE_IDS))
            queue, spawned = FakeQueue(), {}
            torch_stub = types.ModuleType("torch")
            mp_stub = types.ModuleType("torch.multiprocessing")
            mp_stub.Manager = lambda: types.SimpleNamespace(Queue=lambda: queue)
            mp_stub.spawn = lambda fn, args, nprocs: spawned.update(nprocs=nprocs, end_time=args[1])
            torch_stub.multiprocessing = mp_stub
            with workdir() as tmp:
                path = tmp / "test.json"
                path.write_text(json.dumps(challenges))
                source = (SRC / "starter.py").read_text().replace(
                    f"{COMP_DIR}/arc-agi_test_challenges.json", str(path))
                self.assertIn(str(path), source)
                with mock.patch.dict(sys.modules, {"torch": torch_stub, "torch.multiprocessing": mp_stub}), \
                        mock.patch.dict(os.environ, {"KAGGLE_IS_COMPETITION_RERUN": "1"}), \
                        mock.patch.object(sys, "argv", ["starter.py", "--end-time", "123.0"]):
                    exec(compile(source, "starter.py", "exec"), {"__name__": "__main__"})
            self.assertEqual(queue.items, sorted(challenges) + [None] * 4)
            self.assertEqual(spawned, {"nprocs": 4, "end_time": 123.0})

    def test_solver_output_format_matches_decoder_parser(self):
        """Static: what arc_solver.py writes (bz2 pickle, file name = subkey `<id>_<i>.<aug>`, dict fields) is what
        arc_decoder.load_decoded_results parses and the notebook's last cell reads, so a mismatch cannot silently give all [[0]]."""
        solver = (SRC / "arc_solver.py").read_text()
        decoder = (SRC / "arc_decoder.py").read_text()
        for needle in ('dir_outputs = "/kaggle/inference_outputs"',
                       'bz2.BZ2File(os.path.join(dir_outputs, subkey), "w")', "pickle.dump(decoded_result, f)",
                       '"beam_score": beam_score', '"score_aug": augmented_scores', '"solution": solution',
                       'bk = subkey.split(".")[0]', "puzzle_ds_multi = puzzle_ds.split_multi_replies()"):
            self.assertIn(needle, solver)
        for needle in ("bz2.BZ2File(os.path.join(store, key))", "pickle.load(f)", 'base_key = key.split(".")[0]',
                       'g["solution"]', 'g["beam_score"]', 'g["score_aug"]'):
            self.assertIn(needle, decoder)
        self.assertIn(f'load_decoded_results("{OUT_DIR}")', final_cell_source())
        self.assertIn("base_id, base_nr = k.split('_')", (SRC / "arc_loader.py").read_text())

    def test_no_hardcoded_ids_outside_commit_branch(self):
        """The 4 smoke ids appear only in starter.py's `if not rerun_mode` branch."""
        for path in SRC.glob("*.py"):
            text = path.read_text()
            for tid in HARDCODED_SMOKE_IDS:
                if tid in text:
                    self.assertEqual(path.name, "starter.py")
                    lines = text.splitlines()
                    idx = next(i for i, line in enumerate(lines) if tid in line)
                    self.assertEqual(lines[idx - 1].strip(), "if not rerun_mode:")
        notebook_text = NOTEBOOK.read_text()
        self.assertEqual([t for t in HARDCODED_SMOKE_IDS if t in final_cell_source()], [])
        self.assertIn("arc-agi_test_challenges.json", notebook_text)


def write_evidence():
    import datetime
    import subprocess
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ForkHiddenShape)
    names = [t.id().rsplit(".", 1)[-1] for t in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    src_sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SRC.glob("*.py"))}
    evidence = {"passed": result.wasSuccessful(),
                "command": ".venv/bin/python tests/test_arc_agi_2_fork_hidden_shape.py --write-evidence",
                "tests": names,
                "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "src_sha256": src_sha,
                "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()}
    out = ROOT / "artifacts/verification/arc-agi-2/fork-perfpatch/hidden-shape-test-v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=1) + "\n")
    print(out)
    return result.wasSuccessful()


if __name__ == "__main__":
    if "--write-evidence" in sys.argv:
        sys.exit(0 if write_evidence() else 1)
    unittest.main()
