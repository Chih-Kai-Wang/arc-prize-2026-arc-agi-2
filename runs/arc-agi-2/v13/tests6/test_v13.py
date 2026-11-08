"""Local tests for the v13 LLM phase: fill-rule units, a hidden-shape rerun (unseen IDs, counts, test-input numbers,
order) against the stub server, the commit measurement path, skip paths, and finalize. No GPU, no network.
  uv run --quiet --no-project --with numpy --with requests python runs/arc-agi-2/v13/tests5/test_v13.py
tests6 = tests/ (v2 suite) pointed at src6 + the C2 flags (prereg-c2-effort-repair-v1.md)."""
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src6"
sys.path.insert(0, str(SRC))
import v13_merge as M  # noqa: E402

PY_TGZ = "cpython-3.12.14+20260924-x86_64-unknown-linux-gnu-install_only.tar.gz"
P1 = [[1, 2], [3, 4]]
P2 = [[5]]
HA1, HA2 = [[7, 7]], [[8]]
PH = [[0]]
C2 = ("--effort-schedule", "medium:20000,medium:32000", "--repair-rounds", "2", "--repair-cap", "8000")


def test_fill_rule():
    e = {"attempt_1": HA1, "attempt_2": HA2}
    assert M.fill_entry(e, []) == e
    assert M.fill_entry(e, [HA2]) == e                       # program agrees with an NVARC attempt: unchanged
    assert M.fill_entry(e, [P1]) == {"attempt_1": P1, "attempt_2": HA1}
    assert M.fill_entry({"attempt_1": PH, "attempt_2": PH}, [P1, P2]) == {"attempt_1": P1, "attempt_2": P2}
    assert M.fill_entry({"attempt_1": PH, "attempt_2": PH}, [P1]) == {"attempt_1": P1, "attempt_2": PH}
    assert M.ranked_program_answers([[P1], [P2], [P2]], 0) == [P2, P1]   # majority first
    assert M.ranked_program_answers([[P1], [P2]], 0) == [P1, P2]         # tie: first verified
    assert M.ranked_program_answers([[[[11]]], [P2]], 0) == [P2]         # invalid grid ignored
    assert M.ranked_program_answers([[P1]], 1) == []                     # missing index ignored
    test = {"a": {"test": [{}, {}]}, "b": {"test": [{}]}}
    base = {"a": [{"attempt_1": HA1, "attempt_2": HA2}, {"attempt_1": PH, "attempt_2": PH}]}  # 'b' missing in base
    merged, changed = M.merge(test, base, {"a": [[P1, P2]], "b": [[P2]]})
    M.validate(test, merged)
    assert merged["a"][0] == {"attempt_1": P1, "attempt_2": HA1} and merged["a"][1]["attempt_1"] == P2
    assert merged["b"][0] == {"attempt_1": P2, "attempt_2": PH} and changed == 3
    print("fill rule ok")


def rand_grid(rng, h=None, w=None):
    h, w = h or rng.randint(1, 6), w or rng.randint(1, 6)
    return [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]


def make_task(rng, identity, n_test):
    train = []
    for _ in range(rng.randint(2, 4)):
        g = rand_grid(rng)
        train.append({"input": g, "output": [list(r) for r in g] if identity else [r[::-1] for r in g][::-1]})
    if not identity:  # make sure the flip differs from the identity on every pair
        for p in train:
            p["input"][0][0], p["input"][-1][-1] = 1, 2
            p["output"] = [r[::-1] for r in p["input"]][::-1]
    return {"train": train, "test": [{"input": rand_grid(rng)} for _ in range(n_test)]}


def setup_inputs(root, rng, n_tasks, with_inputs=True, eval_too=False):
    comp = root / "input/competitions/arc-prize-2026-arc-agi-2"
    comp.mkdir(parents=True)
    ids = [f"{rng.getrandbits(32):08x}" for _ in range(n_tasks)]
    tasks = {k: make_task(rng, identity=(i % 3 != 2), n_test=1 + (i % 4 == 1) + (i % 5 == 3)) for i, k in enumerate(ids)}
    order = list(ids)
    rng.shuffle(order)
    (comp / "arc-agi_test_challenges.json").write_text(json.dumps({k: tasks[k] for k in order}))
    if eval_too:
        (comp / "arc-agi_evaluation_challenges.json").write_text(json.dumps(tasks))
        sol = {k: [t["input"] if (i % 3 != 2) else [r[::-1] for r in t["input"]][::-1] for t in tasks[k]["test"]]
               for i, k in enumerate(ids)}
        (comp / "arc-agi_evaluation_solutions.json").write_text(json.dumps(sol))
    if with_inputs:
        (root / "input/datasets/wh").mkdir(parents=True)
        (root / "input/datasets/wh/vllm-0.19.0-cp38-abi3-manylinux_2_31_x86_64.whl").write_text("x")
        (root / "input/datasets/py").mkdir(parents=True)
        (root / f"input/datasets/py/{PY_TGZ}").write_text("x")
        (root / "input/models/q38").mkdir(parents=True)
        (root / "input/models/q38/layers-0.safetensors").write_text("x")
    return tasks, ids


# Some hosts (agent sandboxes) deny bind() outright, so the stub server cannot listen. There the phase runs with this
# shim instead: it routes the local HTTP calls inside the v13_llm process to fake_vllm.chat, the same reply policy.
SHIM = '''import json, os, runpy, sys
from urllib.parse import urlsplit

import requests

_FAKE = [None]


def fake():
    if _FAKE[0] is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("fake_vllm", os.environ["V13_FAKE_VLLM"])
        _FAKE[0] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_FAKE[0])
    return _FAKE[0]


class R:
    def __init__(self, status=0, text="", blocks=None):
        self.status_code, self.text, self.blocks, self._closed = status, text, blocks or [], False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        self._closed = True

    def iter_lines(self, chunk_size=None):
        for block in self.blocks:
            if self._closed:
                return
            for line in block.split("\\n"):
                yield line.encode()


def _json(status, obj):
    return R(status, json.dumps(obj))


def _chat(body):
    r = R()
    status, kind, payload = fake().chat(body, closed=lambda: r._closed)
    r.status_code = status
    if kind == "json":
        r.text = json.dumps(payload)
    else:
        r.blocks = payload
    return r


def _resp(method, url, body):
    path = urlsplit(url).path
    if path == "/v1/models":
        return _json(200, {"data": [{"id": "m"}]})
    if path == "/metrics":
        return R(200, "vllm:generation_tokens_total 123.0\\nvllm:num_requests_running 2.0\\n")
    if path == "/tokenize":
        n = sum(len(m.get("content", "")) for m in (body or {}).get("messages", [])) // 3
        return _json(200, {"count": n, "tokens": []})
    if path == "/v1/chat/completions":
        return _chat(body or {})
    return _json(404, {})


_orig = requests.request, requests.post


def _request(method, url, **kw):
    if url.startswith("http://127.0.0.1:"):
        return _resp(method, url, kw.get("json"))
    return _orig[0](method, url, **kw)


def _post(url, **kw):
    if url.startswith("http://127.0.0.1:"):
        return _resp("POST", url, kw.get("json"))
    return _orig[1](url, **kw)


requests.request, requests.post = _request, _post
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''


def can_bind():
    if can_bind.ok is None:
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            can_bind.ok = True
        except OSError:
            can_bind.ok = False
        finally:
            s.close()
    return can_bind.ok


can_bind.ok = None


def run_phase(root, rerun, seconds=60, extra=(), env_extra=None):
    env = dict(os.environ, V13_PORT=str(18000 + random.randint(0, 999)), **(env_extra or {}))
    env.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    if rerun:
        env["KAGGLE_IS_COMPETITION_RERUN"] = "1"
    args = ["--end-time", str(time.time() + seconds), "--work",
            str(root / "work"), "--outputs", str(root / "outputs"), "--input-root", str(root / "input"), "--tmp",
            str(root / "tmp"), "--fake-server", str(HERE / "fake_vllm.py"), "--need-seconds", "5", "--abort-margin",
            "5", "--stop-margin", "10", "--conc", "4", *extra]
    cmd = [sys.executable, str(SRC / "v13_llm.py"), *args]
    if not can_bind():
        shim = root / "fake-shim.py"
        if not shim.exists():
            shim.write_text(SHIM)
        env.update(V13_FAKE_VLLM=str(HERE / "fake_vllm.py"), FAKE_NO_BIND="1")
        cmd = [sys.executable, str(shim), str(SRC / "v13_llm.py"), *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 120, env=env)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return json.loads((root / "work/v13-llm.json").read_text()), p.stdout


def base_submission(test, rng):
    base = {}
    for k, t in test.items():
        base[k] = [{"attempt_1": rand_grid(rng), "attempt_2": PH} for _ in t["test"]]
    return base


def rerun_hidden_shape(extra=()):
    rng = random.Random(7)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        tasks, ids = setup_inputs(root, rng, 11)
        (root / "work").mkdir()
        (root / "outputs").mkdir()
        test = json.loads((root / "input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json").read_text())
        base = base_submission(test, rng)
        k0 = ids[0]  # NVARC already has the identity answer for one output: must stay unchanged
        base[k0][0]["attempt_2"] = test[k0]["test"][0]["input"]
        (root / "work/submission.json").write_text(json.dumps(base))
        S, out = run_phase(root, rerun=True, extra=extra)
        assert S["skipped"] is None, S
        sub = json.loads((root / "work/submission.json").read_text())
        M.validate(test, sub)
        assert json.loads((root / "work/v13-base-submission.json").read_text()) == base
        rows = [json.loads(x) for x in (root / "work/v13-programs.jsonl").read_text().splitlines()]
        ver = {r["task"] for r in rows if r.get("verified")}
        assert ver and all(ids.index(k) % 3 != 2 for k in ver), "only identity tasks can verify"
        assert any(r.get("source") == "reasoning" for r in rows if r.get("verified")) or \
            not any(r.get("finish_reason") == "length" for r in rows), "truncation salvage never verified"
        for k, t in test.items():
            for i, q in enumerate(t["test"]):
                e, b = sub[k][i], base[k][i]
                if k in ver:
                    assert q["input"] in (e["attempt_1"], e["attempt_2"]), (k, i)
                    if k == k0 and i == 0:
                        assert e == b
                    else:
                        assert e == {"attempt_1": q["input"], "attempt_2": b["attempt_1"]}, (k, i, e, b)
                else:
                    assert e == b, (k, i)
        # later reps only for tasks without a verified program; each verified task verified exactly once
        assert all(sum(1 for r in rows if r["task"] == k and r.get("verified")) == 1 for k in ver)
        assert S["order"]["n"] == len(test) and S["summary"]["verified_tasks"] == len(ver)
        rec = M.finalize(str(root / "work"), str(root / "input/competitions/arc-prize-2026-arc-agi-2/"
                                                 "arc-agi_test_challenges.json"))
        assert rec["finalize"] == "merged" and json.loads((root / "work/submission.json").read_text()) == sub
        print(f"rerun hidden-shape ok: {len(test)} tasks, {sum(len(t['test']) for t in test.values())} outputs, "
              f"{len(rows)} jobs, verified tasks {len(ver)}, from reasoning "
              f"{S['summary']['verified_from_reasoning']}")


def test_rerun_hidden_shape():
    rerun_hidden_shape()


def test_c2_hidden_shape():
    rerun_hidden_shape(C2)


def test_commit_measurement():
    rng = random.Random(11)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        tasks, ids = setup_inputs(root, rng, 6, eval_too=True)
        (root / "work").mkdir()
        (root / "outputs").mkdir()
        (root / "outputs" / f"{ids[1]}_0.0").write_text("not a pickle")  # a smoke task with an unreadable store file
        fallback = {k: [{"attempt_1": PH, "attempt_2": PH} for _ in t["test"]] for k, t in tasks.items()}
        (root / "work/submission.json").write_text(json.dumps(fallback))
        S, _ = run_phase(root, rerun=False)
        assert S["skipped"] is None and S["commit"] and S["cap"] == 64000, S
        assert S["order"]["n"] == 1 and S["order"]["first"] == [ids[1]]  # smoke task only (keys30 ids absent here)
        assert json.loads((root / "work/submission.json").read_text()) == fallback  # test-file fallback untouched
        assert not (root / "work/v13-base-submission.json").exists()
        if S["summary"]["verified"]:
            ev = json.loads((root / "work/eval-v13-submission.json").read_text())
            M.validate(json.loads((root / "input/competitions/arc-prize-2026-arc-agi-2/"
                                           "arc-agi_evaluation_challenges.json").read_text()), ev)
            assert S["summary"]["verified_correct"] == S["summary"]["verified"]
        print("commit measurement ok", S["summary"])


def test_skip_paths():
    rng = random.Random(3)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup_inputs(root, rng, 4, with_inputs=False)
        (root / "work").mkdir()
        test = json.loads((root / "input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json").read_text())
        base = base_submission(test, rng)
        (root / "work/submission.json").write_text(json.dumps(base))
        S, _ = run_phase(root, rerun=True)
        assert S["skipped"].startswith("inputs missing"), S
        assert json.loads((root / "work/submission.json").read_text()) == base
        rec = M.finalize(str(root / "work"), str(root / "input/competitions/arc-prize-2026-arc-agi-2/"
                                                 "arc-agi_test_challenges.json"))
        assert rec["finalize"] == "merged" and json.loads((root / "work/submission.json").read_text()) == base
        S, _ = run_phase(root, rerun=True, extra=("--need-seconds", "99999"))
        assert S["skipped"].startswith("not enough time"), S
        (root / "work/submission.json").write_text("{broken")
        (root / "work/v13-base-submission.json").write_text("{broken too")
        rec = M.finalize(str(root / "work"), str(root / "input/competitions/arc-prize-2026-arc-agi-2/"
                                                 "arc-agi_test_challenges.json"))
        assert rec["finalize"].startswith("error") and rec["kept"].startswith("nothing valid")
        print("skip paths ok")


def test_abort_salvage(mode="1"):
    """Streams still running (mode 1) or stalled without any byte (mode 'stall', a preempted request) at the abort
    time are cut; their reasoning drafts are verified and merged, and the phase returns before its end time."""
    rng = random.Random(5)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        tasks, ids = setup_inputs(root, rng, 5)
        (root / "work").mkdir()
        (root / "outputs").mkdir()
        test = json.loads((root / "input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json").read_text())
        base = base_submission(test, rng)
        (root / "work/submission.json").write_text(json.dumps(base))
        t = time.time()
        S, _ = run_phase(root, rerun=True, seconds=40, extra=("--conc", "8"), env_extra={"FAKE_SLOW": mode})
        rows = [json.loads(x) for x in (root / "work/v13-programs.jsonl").read_text().splitlines()]
        assert len(rows) == 5 and all(r["state"] == "aborted" for r in rows), rows
        ver = {r["task"] for r in rows if r.get("verified")}
        assert ver == {k for i, k in enumerate(ids) if i % 3 != 2}, (ver, ids)
        sub = json.loads((root / "work/submission.json").read_text())
        M.validate(test, sub)
        assert all(sub[k][i]["attempt_1"] == q["input"] for k in ver for i, q in enumerate(test[k]["test"]))
        assert time.time() - t < 60, "phase must return before its end time"
        print(f"abort salvage ({mode}) ok", S["summary"], S.get("aborted_streams"))


def test_invert_aug_on_eval():
    """Zero-GPU check of the answer path: for every public-eval task and the first 6 rep salts, the inverse of the
    augmentation maps each augmented train output back to the original exactly."""
    import arc_llm as L
    data = HERE.parents[3] / "data/arc-agi-2/official-v1/arc-agi_evaluation_challenges.json"
    if not data.exists():
        print("invert_aug: eval data not found locally, skipped")
        return
    tasks = json.loads(data.read_text())
    n = 0
    for k, t in tasks.items():
        for rep in range(6):
            aug = L.augmentation(k, salt=f"v13-r{rep}")
            at = L.augment_task(t, aug)
            for a, o in zip(at["train"], t["train"]):
                assert L.invert_aug(a["output"], aug) == o["output"], (k, rep)
                n += 1
    print(f"invert_aug ok on {len(tasks)} eval tasks x 6 salts ({n} grids)")


def test_sandbox():
    import progsynth as P
    task = {"train": [{"input": [[1, 2]], "output": [[1, 2]]}], "test": [{"input": [[3]]}]}
    ident = "def transform(g):\n    return g\n"
    assert P.check(ident, task)["verified"]
    flood = "def transform(g):\n    for _ in range(3000000):\n        print('x' * 100)\n    return g\n"
    res, err = P.run_program(flood, [[[1]]], timeout=10)  # a print flood never reaches the parent
    assert (res and res[0]["ok"] == [[1]]) or err == "timeout", (res, err)
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            w = "import os\nopen('pwned.txt', 'w').write('x')\ndef transform(g):\n    return g\n"
            P.run_program(w, [[[1]]])
            assert not os.path.exists(os.path.join(d, "pwned.txt")), "child must not write into the caller's cwd"
        finally:
            os.chdir(cwd)
    fork = ("import os, time\nif os.fork() == 0:\n    time.sleep(120)\n    os._exit(0)\n"
            "def transform(g):\n    time.sleep(120)\n    return g\n")
    t = time.time()
    res, err = P.run_program(fork, [[[1]]], timeout=3)
    assert err == "timeout" and time.time() - t < 15, (res, err)
    res, err = P.run_program("import sys\nsys.exit(3)\ndef transform(g):\n    return g\n", [[[1]]])
    assert res is None and "SystemExit" in err, (res, err)
    res, err = P.run_program("def transform(g):\n    raise SystemExit(1)\n", [[[1]]])
    assert res and "err" in res[0], (res, err)
    print("sandbox ok")


def test_guards():
    sys.argv = ["x"]
    import v13_llm as V
    task = {"train": [{"input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "output": [[9, 8, 7], [6, 5, 4], [3, 2, 1]]}],
            "test": [{"input": [[2, 2, 2], [3, 3, 3], [4, 4, 4]]}]}
    hard = "def transform(g):\n    if g[0][0] == 1:\n        return [[9, 8, 7], [6, 5, 4], [3, 2, 1]]\n    return g\n"
    assert V.suspicious(hard, task, [[[4, 4, 4], [3, 3, 3], [2, 2, 2]]]) == "train output hard-coded"
    rot = "def transform(g):\n    return [r[::-1] for r in g][::-1]\n"
    assert V.suspicious(rot, task, [[[4, 4, 4], [3, 3, 3], [2, 2, 2]]]) is None
    assert V.suspicious(rot, task, [[[2, 2, 2], [3, 3, 3], [4, 4, 4]]]) == "identity on test only"
    assert V.suspicious(rot, task, [[[9, 8, 7], [6, 5, 4], [3, 2, 1]]]) == "test answer copies a train output"
    small = {"train": [{"input": [[1, 1]], "output": [[3]]}, {"input": [[2]], "output": [[4]]}],
             "test": [{"input": [[5, 5]]}]}
    assert V.suspicious("def transform(g):\n    return [[3]]\n", small, [[[3]]]) is None  # tiny outputs may repeat
    print("guards ok")


def test_invalid_base_skips():
    rng = random.Random(9)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup_inputs(root, rng, 3)
        (root / "work").mkdir()
        (root / "outputs").mkdir()
        (root / "work/submission.json").write_text(json.dumps({"wrong": []}))
        S, _ = run_phase(root, rerun=True)
        assert S["skipped"].startswith("NVARC base unreadable/invalid"), S
        assert not (root / "work/v13-base-submission.json").exists()
        assert json.loads((root / "work/submission.json").read_text()) == {"wrong": []}
        print("invalid base skip ok")


def test_c2_schedule_and_repair():
    """Per-rep effort/cap schedule + one repair round: a stub seed%4==2 job (wrong program, stop) is repaired with
    execution feedback to the identity program; the fake server's log shows the caps and the effort actually sent."""
    rng = random.Random(17)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        tasks, ids = setup_inputs(root, rng, 11)
        (root / "work").mkdir()
        (root / "outputs").mkdir()
        test = json.loads((root / "input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json").read_text())
        base = base_submission(test, rng)
        (root / "work/submission.json").write_text(json.dumps(base))
        logp = root / "fake-chat.jsonl"
        S, out = run_phase(root, rerun=True, extra=C2, env_extra={"FAKE_LOG": str(logp)})
        assert S["skipped"] is None, S
        assert S["effort_schedule"] == [["medium", 20000], ["medium", 32000]], S
        assert S["effort_smoke"], S
        rows = [json.loads(x) for x in (root / "work/v13-programs.jsonl").read_text().splitlines()]
        rep = [r for r in rows if str(r.get("source") or "").startswith("repair")]
        assert rep and all(r.get("repair") and r.get("code") for r in rep), rows
        assert all(r.get("effort") == "medium" for r in rows), rows
        log = [json.loads(x) for x in logp.read_text().splitlines()]
        assert log and all(e["reasoning_effort"] == "medium" for e in log if e["stream"]), log
        seed_rep = {zlib.crc32(f"v13:{k}:{r}".encode()) & 0x7FFFFFFF: r for k in test for r in range(6)}
        first = [e for e in log if e["stream"] and not e["repair"]]
        assert first and all(e["seed"] in seed_rep for e in first), first
        assert all(e["max_tokens"] <= (20000 if seed_rep[e["seed"]] == 0 else 32000) for e in first), first
        fix = [e for e in log if e["repair"]]
        assert fix and all(e["max_tokens"] <= 8000 for e in fix), fix
        assert S["summary"]["repaired"] >= 1, S["summary"]
        print(f"C2 schedule/repair ok: {len(rows)} rows, {len(rep)} repaired, {len(fix)} repair requests, "
              f"{S['summary']}")


def test_c2_bad_flags():
    """Every invalid C2 value is rejected before any work, with exit code 2 and a `bad C2 flag` message."""
    rng = random.Random(23)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup_inputs(root, rng, 3)
        (root / "work").mkdir()
        (root / "outputs").mkdir()
        for extra in (("--effort-schedule", "high:20000"), ("--effort-schedule", "medium:100"),
                      ("--effort-schedule", "medium"), ("--repair-rounds", "9"), ("--repair-cap", "50")):
            env = dict(os.environ, V13_PORT=str(20000 + random.randint(0, 999)))
            env.pop("KAGGLE_IS_COMPETITION_RERUN", None)
            cmd = [sys.executable, str(SRC / "v13_llm.py"), "--end-time", str(time.time() + 30), "--work",
                   str(root / "work"), "--outputs", str(root / "outputs"), "--input-root", str(root / "input"),
                   "--tmp", str(root / "tmp"), "--fake-server", str(HERE / "fake_vllm.py"), *extra]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
            out = p.stdout + p.stderr
            assert p.returncode != 0, (extra, out)
            assert "bad C2 flag" in out, (extra, out)
        print("C2 bad flags ok")


def test_c2_effort_gate():
    """A server that rejects reasoning_effort medium skips the phase at the in-kernel effort smoke: no job runs."""
    rng = random.Random(29)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup_inputs(root, rng, 5)
        (root / "work").mkdir()
        (root / "outputs").mkdir()
        test = json.loads((root / "input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json").read_text())
        base = base_submission(test, rng)
        (root / "work/submission.json").write_text(json.dumps(base))
        S, out = run_phase(root, rerun=True, extra=C2, env_extra={"FAKE_REJECT_EFFORT": "medium"})
        assert S["skipped"] and "effort medium rejected" in S["skipped"], S
        rows = root / "work/v13-programs.jsonl"
        assert not rows.exists() or not rows.read_text().strip(), rows
        print("C2 effort gate ok:", S["skipped"])


if __name__ == "__main__":
    test_fill_rule()
    test_invert_aug_on_eval()
    test_sandbox()
    test_guards()
    test_invalid_base_skips()
    test_rerun_hidden_shape()
    test_commit_measurement()
    test_skip_paths()
    test_abort_salvage()
    test_abort_salvage("stall")
    test_c2_schedule_and_repair()
    test_c2_bad_flags()
    test_c2_effort_gate()
    test_c2_hidden_shape()
    print("ALL OK")
