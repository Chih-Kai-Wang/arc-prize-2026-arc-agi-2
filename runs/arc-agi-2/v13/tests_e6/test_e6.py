"""E6 checks (zero GPU): the job list is keys30 x reps (1, 2) interleaved exactly as E3's seed jobs (same seeds, no
rep-0 tail); the built notebook's v13_llm call is `--jobs e6-jobs.json --cap 96000` with no forcing; on the tests3 fake
server that call runs each listed job once with cap 96000; the KV formula gives conc 10 at 96k; the launch gate accepts
the build and rejects a tampered job list. Kept outside tests3 so the v3 audit's test set is unchanged.
  uv run --quiet --no-project --with numpy --with requests --with nbformat python runs/arc-agi-2/v13/tests_e6/test_e6.py [SLUG]"""
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests3"))
sys.path.insert(0, str(HERE.parent))
import test_v13 as T  # noqa: E402  (setup_inputs, run_phase on src3 + fake_vllm)
import build_e3 as E3  # noqa: E402
import build_e6 as E6  # noqa: E402

SLUG = sys.argv[1] if len(sys.argv) > 1 else "arc-2026-v13-e6-v4seeds-v1"


def test_job_list_and_conc():
    jobs = E6.jobs()
    e3_seed_jobs = [j for j in E3.jobs() if j[1] in E3.SEED_REPS]
    assert jobs == e3_seed_jobs and len(jobs) == 60 and len({tuple(j) for j in jobs}) == 60, jobs
    assert [r for _, r in jobs[:4]] == [1, 2, 1, 2] and jobs[0][0] == jobs[1][0]
    kv = 331200  # E3 / E4 / v4-c96 server log "GPU KV cache size: 331,200 tokens"
    conc = lambda cap: max(8, min(32, int(kv / (0.35 * cap + 6000)) + 2))  # v13_llm.py formula (src3)
    assert conc(E6.CAP) == 10
    assert E6.E6_SECONDS + E3.B.BOOT_OVERHEAD_SECONDS < 43200  # Kaggle 12 h session limit


def test_fake_run_cap96k():
    rng = random.Random(61)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        tasks, ids = T.setup_inputs(root, rng, 4, eval_too=True)
        (root / "work").mkdir()
        jobs = [[k, r] for k in ids for r in E6.SEED_REPS]
        path = root / "e6-jobs.json"
        path.write_text(json.dumps(jobs))
        S, _ = T.run_phase(root, rerun=False, seconds=90, extra=("--jobs", str(path), "--cap", str(E6.CAP)))
        assert S["skipped"] is None and S["commit"] and S["cap"] == 96000 and S["force_at"] == 0, S
        assert S["order"]["n"] == len(jobs) and S["order"]["requeue"] is False
        rows = [json.loads(x) for x in (root / "work/v13-programs.jsonl").read_text().splitlines()]
        assert sorted([r["task"], r["rep"]] for r in rows) == sorted(jobs), rows
        assert all(int(r["cap"]) == 96000 for r in rows), [r["cap"] for r in rows]
        print(f"fake cap-96k two-seed run ok: {len(rows)} jobs, verified {sum(bool(r.get('verified')) for r in rows)}")


def test_notebook_and_gate():
    folder = E6.HERE / f"upload-{SLUG}"
    nb = json.loads((folder / f"{SLUG}.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    calls = [t for t in code if "python v13_llm.py" in t and not t.startswith("%%writefile")]
    assert calls == [E6.llm_command()] and "--cap 96000" in calls[0] and "--force-at" not in calls[0], calls
    assert any("KAGGLE_IS_COMPETITION_RERUN" in t and "assert not" in t for t in code)
    jobs = [t for t in code if t.startswith("%%writefile e6-jobs.json")]
    assert len(jobs) == 1 and json.loads(jobs[0].split("\n", 1)[1]) == E6.jobs()
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False and meta["is_private"] is True and meta["machine_shape"] == "NvidiaL4"
    sys.path.insert(0, str(E6.ROOT))
    try:
        import launch_v13 as L  # noqa: E402
    except ImportError:  # public release: the Kaggle launch/quota layer (launch_v13.py) is not published
        print("SKIP gate checks: launch_v13 not available")
        return
    prep, _ = L.gate(folder)
    assert prep["stage"] == "v13-e6" and prep["planned_commit_wall_seconds"] == E6.E6_SECONDS + 600
    with tempfile.TemporaryDirectory() as d:  # tampered job list -> gate refuses
        bad = Path(d) / folder.name
        shutil.copytree(folder, bad)
        p = bad / f"{SLUG}.ipynb"
        text = p.read_text()
        assert ', 2], [\\"' in text
        p.write_text(text.replace(', 2], [\\"', ', 3], [\\"', 1))  # one job's rep 2 -> 3, still valid JSON
        json.loads(p.read_text())
        try:
            L.gate(bad)
        except BaseException as e:  # cloud.need raises
            assert "notebook changed after build" in str(e), e
            print("tampered build refused:", e)
        else:
            raise AssertionError("gate accepted a tampered notebook")


if __name__ == "__main__":
    test_job_list_and_conc()
    test_notebook_and_gate()
    test_fake_run_cap96k()
    print("E6 tests passed")
