"""E4 checks (zero GPU): the built notebook's v13_llm call (--jobs e4-jobs.json --cap 96000) on the tests3 fake server
runs each listed job once at rep 3 with cap 96000 (prompt-bounded), the KV formula gives conc 10 at 96k for the E3
server's KV, and the job list is the pre-registered one. Kept outside tests3 so the v3 audit's test set is unchanged.
  uv run --quiet --no-project --with numpy --with requests --with nbformat python runs/arc-agi-2/v13/tests_e4/test_e4.py"""
import json
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests3"))
sys.path.insert(0, str(HERE.parent))
import test_v13 as T  # noqa: E402  (setup_inputs, run_phase on src3 + fake_vllm)
import build_e4 as E4  # noqa: E402

PREREG_JOBS = [["e3721c99", 3], ["3e6067c3", 3], ["332f06d7", 3], ["faa9f03d", 3], ["f931b4a8", 3], ["6e4f6532", 3],
               ["7b0280bc", 3], ["13e47133", 3], ["142ca369", 3], ["e12f9a14", 3]]


def test_job_list_and_conc():
    assert E4.jobs() == PREREG_JOBS, E4.jobs()
    kv = 331200  # E3 server log "GPU KV cache size: 331,200 tokens"
    conc = lambda cap: max(8, min(32, int(kv / (0.35 * cap + 6000)) + 2))  # v13_llm.py formula (src3 L892)
    assert (conc(64000), conc(80000), conc(96000)) == (13, 11, 10)


def test_fake_run_cap96k():
    rng = random.Random(41)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        tasks, ids = T.setup_inputs(root, rng, 6, eval_too=True)
        (root / "work").mkdir()
        jobs = [[k, 3] for k in ids]
        path = root / "e4-jobs.json"
        path.write_text(json.dumps(jobs))
        S, _ = T.run_phase(root, rerun=False, seconds=90, extra=("--jobs", str(path), "--cap", str(E4.CAP)))
        assert S["skipped"] is None and S["commit"] and S["cap"] == 96000 and S["force_at"] == 0, S
        assert S["order"]["n"] == len(jobs) and S["order"]["requeue"] is False
        rows = [json.loads(x) for x in (root / "work/v13-programs.jsonl").read_text().splitlines()]
        assert sorted([r["task"], r["rep"]] for r in rows) == sorted(jobs), rows
        assert all(int(r["cap"]) == 96000 for r in rows), [r["cap"] for r in rows]
        print(f"fake cap-96k run ok: {len(rows)} jobs, verified {sum(bool(r.get('verified')) for r in rows)}")


def test_notebook_call():
    nb = json.loads((E4.HERE / "upload-arc-2026-v13-e4-cap96k-v1/arc-2026-v13-e4-cap96k-v1.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    calls = [t for t in code if "python v13_llm.py" in t and not t.startswith("%%writefile")]
    assert calls == [E4.llm_command()] and "--cap 96000" in calls[0] and "--force-at" not in calls[0], calls
    assert any("KAGGLE_IS_COMPETITION_RERUN" in t and "assert not" in t for t in code)
    jobs = [t for t in code if t.startswith("%%writefile e4-jobs.json")]
    assert len(jobs) == 1 and json.loads(jobs[0].split("\n", 1)[1]) == PREREG_JOBS


if __name__ == "__main__":
    test_job_list_and_conc()
    test_notebook_call()
    test_fake_run_cap96k()
    print("E4 tests passed")
