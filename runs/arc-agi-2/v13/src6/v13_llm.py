"""v13 LLM phase (runs after NVARC pass A has been assembled into submission.json): serve Qwen3.8-27B-FP8 with vLLM
0.19.0 from a Python 3.12 venv (python-build-standalone + the cp312 wheelhouse; the NVARC Python 3.11 stack is never
touched) and run the program-synthesis channel: write `transform(grid)`, verify it on every train pair of an augmented
copy of the task, and when it passes put its test answer into attempt_1 (v13_merge fill rule). Every verified program
rewrites the submission atomically, so a hard stop at any moment leaves a valid file.

Any failure before the first job (time, inputs missing, GPUs busy, install, server) skips the phase and keeps NVARC's
submission. Rerun: tasks from the competition test file only, unsure-first by NVARC pass-A confidence. Commit: the
4 NVARC smoke tasks + the 30 public-eval keys30 tasks, scored against the evaluation solutions (dev measurement).
  python v13_llm.py --end-time T   (T = epoch seconds by which this phase must have returned)"""
import argparse
import glob
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arc_llm as L  # noqa: E402
import progsynth as P  # noqa: E402
import v13_merge as M  # noqa: E402

T0 = time.time()
A = None
S = {"version": "v13-llm-v1", "t0": T0, "phases": {}, "skipped": None}
LOCK = threading.Lock()
RESULTS = {}           # task -> list of verified answer lists (original frame)
STOP = threading.Event()
ACTIVE = {}          # open streaming responses; closed at the abort time (a preempted stream sends nothing)
SERVER = [None]        # the server process, for the last-resort kill after the abort time
PORT = int(os.environ.get("V13_PORT", "8000"))  # test seam; production 8000
MAX_LEN = 131072
PY_TGZ = "cpython-3.12.14+20260924-x86_64-unknown-linux-gnu-install_only.tar.gz"
PY_SHA = "5eae8cf79dd47fc2496a4fccc892936be831ce7a84d984b2299dfb1cdb592682"
PYPI = {  # sha256 of the official PyPI wheels (pypi.org JSON API, 2026-10-12; same pins as P0/P1)
    "vllm-0.19.0-cp38-abi3-manylinux_2_31_x86_64.whl":
        "2d0e5fae45367bdbf111fcad68f4c0f8fdddd2f2fb643e52f0f2daebef7b41cf",
    "torch-2.10.0-3-cp312-cp312-manylinux_2_28_x86_64.whl":
        "98c01b8bb5e3240426dcde1446eed6f40c778091c8544767ef1168fc663a05a6",
}
CONFIDENT = 5.0        # arc_v6_config.PASS_B_CONFIDENT: kgmon top-1 score counted as sure (v9)
MAX_REPS = 6           # augmentation repetitions per task; rep r >= 1 only for tasks without a verified program
EFFORTS = ("xhigh", "medium", "low")


def el():
    return round(time.time() - T0, 1)


def log(msg):
    print(f"[v13 {el():8.1f}s] {msg}", flush=True)


def save():
    with LOCK:
        text = json.dumps(S, default=str)
        tmp = os.path.join(A.work, "v13-llm.json.tmp")
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, os.path.join(A.work, "v13-llm.json"))


def append(name, row):
    with LOCK, open(os.path.join(A.work, name), "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def run(cmd, timeout=None, **kw):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
        return p.returncode, p.stdout + p.stderr
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def find(pattern):
    return sorted(glob.glob(os.path.join(A.input_root, "**", pattern), recursive=True))


class Skip(Exception):
    pass


def parse_schedule(text):
    """`EFFORT:CAP[,EFFORT:CAP...]` -> [[effort, cap], ...] or None (bad flag value)."""
    sched = []
    for part in text.split(","):
        effort, sep, cap = part.partition(":")
        if not sep or effort not in EFFORTS or not cap.isdigit() or not 2000 <= int(cap) <= 120000:
            return None
        sched.append([effort, int(cap)])
    return sched or None


def budget(seconds):
    """Timeout for one setup step: never past the setup deadline (server must be ready by then); skip when < 30 s."""
    left = A.setup_deadline - time.time()
    if left < 30:
        raise Skip(f"setup time exhausted ({round(left)}s left)")
    return min(seconds, left)


def venv_env():
    """Environment for every Python 3.12 venv process (pip, checks, server): nothing from the NVARC 3.11 stack leaks in
    (PYTHONPATH/HOME, user site, PIP_*), and the CUDA toolkit dirs do not shadow the torch wheel's bundled CUDA libs."""
    drop = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP", "PYTHONUSERBASE", "CONDA_PREFIX")
    env = {k: v for k, v in os.environ.items() if k not in drop and not k.startswith("PIP_")}
    env["PYTHONNOUSERSITE"] = "1"
    parts = [x for x in env.get("LD_LIBRARY_PATH", "").split(":") if x and not x.startswith("/usr/local/cuda")]
    libcuda = "/usr/local/nvidia/lib64/libcuda.so"
    if os.path.exists(libcuda):  # same libcuda pin as P0/P1 (public Apache-2.0 Qwen3.8 notebook)
        parts = ["/usr/local/nvidia/lib64"] + [x for x in parts if x != "/usr/local/nvidia/lib64"]
        env["LD_PRELOAD"] = libcuda
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


# --- 1. inputs, GPUs, install ------------------------------------------------------------------------------------------

def phase_inputs():
    ph = S["phases"]["inputs"] = {}
    wheels = find("vllm-0.19.0-*.whl")
    tgz = find(PY_TGZ)
    model = [os.path.dirname(p) for p in find("layers-0.safetensors")]
    ph.update(wheels=wheels[:1], py_tgz=tgz[:1], model=model[:2])
    if not (wheels and tgz and len(model) == 1):
        raise Skip(f"inputs missing: wheels={len(wheels)} py={len(tgz)} model={len(model)}")
    return Path(wheels[0]).parent, tgz[0], Path(model[0])


def gpu_used():
    rc, out = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=60)
    vals = [int(x) for x in out.split() if x.strip().isdigit()] if rc == 0 else []
    return vals


def phase_gpus():
    """NVARC workers must be gone: all GPUs near-empty within 5 min, else kill orphan compute processes, else skip."""
    ph = S["phases"]["gpus"] = {}
    for attempt in range(2):
        end = min(time.time() + (300 if attempt == 0 else 120), A.setup_deadline)
        while time.time() < end:
            used = gpu_used()
            if A.fake_server or (used and all(u < 1500 for u in used)):
                ph.update(free_after=el(), used=used, attempt=attempt)
                return
            time.sleep(10)
        rc, out = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], timeout=60)
        pids = [int(x) for x in out.split() if x.strip().isdigit()] if rc == 0 else []
        mine = ancestors()
        killed = []
        for pid in pids:  # only NVARC leftovers: container PIDs whose command line is a starter/multiprocessing child
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            except OSError:
                continue
            if pid not in mine and ("multiprocessing" in cmd or "starter.py" in cmd):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
                except Exception:  # noqa: BLE001
                    pass
        rc2, _ = run(["pkill", "-9", "-f", "multiprocessing.spawn"], timeout=30)  # orphan NVARC workers/manager
        ph.setdefault("killed", []).append({"pids": killed, "pkill_rc": rc2, "used": gpu_used()})
        save()
    raise Skip(f"GPUs not released: {gpu_used()}")


def ancestors():
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except Exception:  # noqa: BLE001
            break
    return pids | {os.getppid()}


def verify_wheels(wheels):
    ph = S["phases"]["wheels"] = {}
    sums = {}
    for line in (wheels / "SHA256SUMS").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]
    files = sorted(p.name for p in wheels.glob("*.whl"))
    bad = [n for n in files if sums.get(n) != sha_file(wheels / n)]
    pins = {n: (sums.get(n) == h and (wheels / n).exists()) for n, h in PYPI.items()}
    ph.update(n_wheels=len(files), mismatched=bad, pypi_pins=pins)
    if bad or not all(pins.values()):
        raise Skip("wheel hash check failed")


def model_hash(model, out):
    """Commit only, informational (P1 already verified this pinned mirror; train-pair verification guards answers):
    sha256 of every weight file vs the HF LFS list. Never blocks the phase."""
    from concurrent.futures import ThreadPoolExecutor
    try:
        ref = json.loads((HERE / "qwen38_fp8_sha256.json").read_text())["files"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {n: pool.submit(sha_file, model / n) for n in ref if (model / n).exists()}
            bad = [n for n, f in futs.items() if f.result() != ref[n][1] or (model / n).stat().st_size != ref[n][0]]
        res = dict(files=len(ref), missing=[n for n in ref if n not in futs], mismatched=bad, done=el())
    except Exception as e:  # noqa: BLE001
        res = {"error": repr(e)[:200]}
    with LOCK:
        out.update(res)


def phase_install(wheels, tgz):
    ph = S["phases"]["install"] = {}
    t = time.time()
    if sha_file(tgz) != PY_SHA:
        raise Skip("python tarball sha256 mismatch")
    root = Path(A.tmp)
    root.mkdir(parents=True, exist_ok=True)
    rc, out = run(["tar", "xzf", tgz, "-C", str(root)], timeout=budget(900))
    if rc != 0:
        raise Skip("python extract failed: " + out[-300:])
    venv = root / "venv"
    rc, out = run([str(root / "python/bin/python3.12"), "-m", "venv", str(venv)], timeout=budget(900), env=venv_env())
    if rc != 0:
        raise Skip("venv failed: " + out[-300:])
    vpy = str(venv / "bin/python")
    verify_wheels(wheels)
    rc, out = run([vpy, "-m", "pip", "install", "--no-index", "--find-links", str(wheels), "--requirement",
                   str(wheels / "requirements.lock")], timeout=budget(2400), env=venv_env())
    ph.update(pip_rc=rc, pip_tail=out[-1500:], pip_seconds=round(time.time() - t))
    if rc != 0:
        raise Skip("pip install failed")
    rc, out = run([vpy, "-c", "import json, torch, vllm, transformers; print(json.dumps({'torch': torch.__version__, "
                   "'cuda_ok': torch.cuda.is_available(), 'gpus': torch.cuda.device_count(), "
                   "'vllm': vllm.__version__, 'transformers': transformers.__version__}))"], timeout=budget(900),
                  env=venv_env())
    ph["import"] = out.strip().splitlines()[-1] if out.strip() else out
    if rc != 0 or '"cuda_ok": true' not in ph["import"]:
        raise Skip("venv torch/vllm import or CUDA check failed")
    rc2, _ = run([vpy, "-c", "import flash_attn"], timeout=budget(300), env=venv_env())
    ph["flash_attn"] = rc2 == 0
    if rc2 != 0:  # P0/P1 fix: vLLM forwards layer_idx to an attention impl that rejects it without flash-attn
        rc3, vroot = run([vpy, "-c", "import os, vllm; print(os.path.dirname(vllm.__file__))"], timeout=budget(300),
                         env=venv_env())
        vroot = vroot.strip().splitlines()[-1] if rc3 == 0 and vroot.strip() else ""
        patched, reverted = [], []
        for f in sorted(glob.glob(vroot + "/**/attention/*.py", recursive=True)) if vroot else []:
            text = Path(f).read_text()
            if "self.impl = impl_cls(" not in text or "extra_impl_args.pop('layer_idx', None)" in text:
                continue
            new = re.sub(r"^(\s*)self\.impl = impl_cls\(",
                         r"\1extra_impl_args.pop('layer_idx', None)\n\1self.impl = impl_cls(", text, count=1, flags=re.M)
            Path(f).write_text(new)
            if run([vpy, "-m", "py_compile", f], timeout=budget(120), env=venv_env())[0] != 0 \
                    or "extra_impl_args" not in text:
                Path(f).write_text(text)
                reverted.append(f)
            else:
                patched.append(f)
        ph.update(attention_patch=patched, attention_patch_reverted=reverted)
    ph["seconds"] = round(time.time() - t)
    save()
    return vpy


# --- 2. server --------------------------------------------------------------------------------------------------------

def server_env():
    return dict(venv_env(), PYTORCH_ALLOC_CONF="expandable_segments:True",
                VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", VLLM_LOGGING_LEVEL="INFO")  # backend: --attention-backend


def http(method, path, body=None, timeout=30):
    import requests
    r = requests.request(method, f"http://127.0.0.1:{PORT}{path}", json=body, timeout=timeout)
    return r.status_code, r.text


def wait_ready(proc, deadline):
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if http("GET", "/v1/models", timeout=5)[0] == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def kill(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def start_server(vpy, model, mtp, eager):
    if A.fake_server:  # test seam: a stub OpenAI-compatible server
        cmd = [sys.executable, A.fake_server, "--port", str(PORT)]
    else:
        cmd = [vpy, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model), "--served-model-name", "m",
               "--host", "127.0.0.1", "--port", str(PORT), "--tensor-parallel-size", "4",
               "--gpu-memory-utilization", "0.93", "--max-model-len", str(MAX_LEN),
               "--max-num-seqs", str(max(48, A.conc)), "--kv-cache-dtype", "fp8", "--attention-backend",
               "TRITON_ATTN", "--reasoning-parser", "qwen3", "--trust-remote-code",
               "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
        if mtp:
            cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": mtp})]
        if eager:
            cmd += ["--enforce-eager"]
    path = os.path.join(A.work, f"v13-server-mtp{mtp}{'-eager' if eager else ''}.log")
    proc = subprocess.Popen(cmd, stdout=open(path, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            text=True, start_new_session=True, env=server_env())
    return proc, path


def smoke():
    body = {"model": "m", "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    code, text = http("POST", "/v1/chat/completions", body, timeout=300)
    return code == 200, text[-300:]


def effort_smoke(efforts):
    """One cheap non-streaming request per distinct effort in the schedule: an unsupported effort is rejected here,
    before any job token is spent."""
    ok = []
    for e in dict.fromkeys(efforts):
        body = {"model": "m", "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 1,
                "temperature": 0, "chat_template_kwargs": {"reasoning_effort": e, "enable_thinking": True}}
        code, text = http("POST", "/v1/chat/completions", body, timeout=300)
        if code != 200:
            raise Skip(f"effort {e} rejected: {text[-200:]}")
        ok.append(e)
    return ok


def phase_server(vpy, model, deadline):
    ph = S["phases"]["server"] = {"attempts": []}
    for mtp, eager in ((2, False), (0, False), (0, True)):
        if time.time() > deadline:
            break
        t = time.time()
        proc, path = start_server(vpy, model, mtp, eager)
        ready = wait_ready(proc, min(t + 25 * 60, deadline))
        att = {"mtp": mtp, "eager": eager, "ready": ready, "startup_seconds": round(time.time() - t)}
        if ready:
            ok, resp = smoke()
            att.update(smoke_ok=ok, smoke=resp)
            text = Path(path).read_text(errors="ignore")
            kv = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
            att["kv_tokens"] = int(kv[-1].replace(",", "")) if kv else None
            if ok:
                ph["attempts"].append(att)
                ph["chosen"] = att
                save()
                log(f"server ready mtp={mtp} eager={eager} in {att['startup_seconds']}s kv={att['kv_tokens']}")
                return proc
        att["log_tail"] = Path(path).read_text(errors="ignore")[-2500:] if os.path.exists(path) else ""
        ph["attempts"].append(att)
        save()
        kill(proc)
        time.sleep(10)
    raise Skip("no server configuration started")


# --- 3. jobs ------------------------------------------------------------------------------------------------------------

_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def code_blocks(text):
    return [b for b in _CODE.findall(text or "") if "def transform" in b]


def prompt_tokens(messages):
    try:
        code, text = http("POST", "/tokenize", {"model": "m", "messages": messages, "chat_template_kwargs":
                                                 {"reasoning_effort": A.effort, "enable_thinking": True}}, timeout=60)
        if code == 200 and json.loads(text).get("count"):
            return json.loads(text)["count"]
    except Exception:  # noqa: BLE001
        pass
    return sum(len(m["content"]) for m in messages) + 300  # characters: an upper bound for these digit grids


def chat(messages, cap, seed, effort=None):
    import requests
    body = {"model": "m", "messages": messages, "max_tokens": cap, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "seed": seed, "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"reasoning_effort": effort or A.effort, "enable_thinking": True}}
    out = {"state": "running", "t_start": el(), "max_tokens": cap}
    content, reasoning, usage, finish = [], [], {}, None
    resp = None
    try:
        with requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, stream=True,
                           timeout=(30, 900)) as r:
            if r.status_code != 200:
                return {**out, "state": "http_error", "error": r.text[-300:], "content": "", "reasoning": "", "t_end": el()}
            resp = r
            with LOCK:
                ACTIVE[id(r)] = r
            if STOP.is_set():
                r.close()
            for raw in r.iter_lines(chunk_size=None):  # as bytes arrive (no 512-byte buffer); split on \r/\n only
                raw = raw.decode("utf-8", "replace")
                if STOP.is_set() or time.time() > A.abort_at:
                    out["state"] = "aborted"
                    break
                if not raw or not raw.startswith("data: ") or raw[6:].strip() == "[DONE]":
                    continue
                ev = json.loads(raw[6:])
                usage = ev.get("usage") or usage
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        content.append(d["content"])
                    rs = d.get("reasoning_content") or d.get("reasoning")
                    if rs:
                        reasoning.append(rs)
                    finish = ch.get("finish_reason") or finish
    except Exception as e:  # noqa: BLE001
        out.update(state="aborted" if STOP.is_set() else "exception", error=repr(e)[-300:])
    finally:
        if resp is not None:
            with LOCK:
                ACTIVE.pop(id(resp), None)
    if STOP.is_set() and out["state"] == "running" and not finish:
        out["state"] = "aborted"
    out.update(t_end=el(), completion_tokens=usage.get("completion_tokens"), prompt_tokens=usage.get("prompt_tokens"),
               finish_reason=finish, content="".join(content), reasoning="".join(reasoning))
    if out["state"] == "running":
        out["state"] = "done" if finish else "incomplete"
    return out


def suspicious(code, task, tests):
    """Cheap guards against programs that pass the train pairs by memorising them (API review 20261013T150520Z #12):
    a train output (>= 9 cells, >= 2 colours) written into the source, a test answer that copies a train output
    (>= 9 cells), or an identity answer on test when no train pair is identity. Returns a reason or None."""
    digits = re.sub(r"\D", "", code)
    outs = [p["output"] for p in task["train"]]
    for o in outs:
        flat = "".join(str(v) for r in o for v in r)
        if len(flat) >= 9 and len(set(flat)) >= 2 and flat in digits:
            return "train output hard-coded"
    for q, t in zip(task["test"], tests):
        if sum(len(r) for r in t) >= 9 and any(t == o for o in outs):
            return "test answer copies a train output"
        if t == q["input"] and not any(p["input"] == p["output"] for p in task["train"]):
            return "identity on test only"
    return None


def run_job(task_id, rep):
    raw = TASKS[task_id]
    aug = L.augmentation(task_id, salt=f"v13-r{rep}")
    task = L.augment_task(raw, aug)
    if A.schedule:
        effort, cap0 = A.schedule[min(rep, len(A.schedule) - 1)]
    else:
        effort, cap0 = A.effort, A.cap
    msgs = [{"role": "user", "content": P.build_prompt(task)}]
    n_prompt = prompt_tokens(msgs)
    cap = min(cap0, MAX_LEN - n_prompt - 512)
    row = {"task": task_id, "rep": rep, "effort": effort, "prompt_tokens_est": n_prompt, "cap": cap}
    if cap < 2000:
        row.update(state="prompt_too_long", verified=False)
        append("v13-programs.jsonl", row)
        return row
    r = chat(msgs, cap, zlib.crc32(f"v13:{task_id}:{rep}".encode()) & 0x7FFFFFFF, effort)
    row.update({k: r.get(k) for k in ("state", "finish_reason", "completion_tokens", "prompt_tokens", "t_start",
                                      "t_end", "error")})
    # candidates: the final answer's code first; then (truncated, aborted or unverified) the last code drafts of the
    # reasoning — zero extra tokens, and train-pair verification guards precision
    cands = [("content", c) for c in code_blocks(r["content"])[-1:]]
    cands += [("reasoning", c) for c in code_blocks(r["reasoning"])[-2:][::-1]]
    row.update(n_content_code=len(code_blocks(r["content"])), n_reasoning_code=len(code_blocks(r["reasoning"])),
               reasoning_len=len(r["reasoning"]), content_len=len(r["content"]), verified=False, tried=[])
    if r.get("finish_reason") != "stop":  # what the model was doing at the cap / abort (input to cap & effort choices)
        row["reasoning_tail"] = r["reasoning"][-2000:]
    seen = set()
    tried = []  # (source, code, chk), the existing candidate order

    def try_code(source, code):
        chk = P.check(code, task)
        why = suspicious(code, task, chk["tests"]) if chk["verified"] else None
        row["tried"].append({"source": source, "verified": chk["verified"], "train_pass": chk["train_pass"],
                             "train_n": chk["train_n"], "error": (chk.get("error") or "")[:200], "rejected": why})
        chk["rejected"] = why
        if chk["verified"] and why is None:
            answers = [L.invert_aug(t, aug) for t in chk["tests"]]
            row.update(verified=True, source=source, answers=answers, code_tail=code[-1500:], code=code[:20000])
            if SOLUTIONS.get(task_id):
                row["correct"] = [a == s for a, s in zip(answers, SOLUTIONS[task_id])]
        return chk

    for source, code in cands:
        if code in seen:
            continue
        seen.add(code)
        chk = try_code(source, code)
        tried.append((source, code, chk))
        if row["verified"]:
            break
    tokens_total = r.get("completion_tokens") or 0
    # repair: feed back the execution mismatches of the best candidate (highest train_pass; guard-rejected programs
    # excluded, their feedback would list no mismatch) and retry; each later round shows the previous round's best
    # new program (Argon pre-push review 10-19: never the same code/feedback twice), history kept
    base = [t for t in tried if not t[2].get("rejected")]
    if not row["verified"] and A.repair_rounds > 0 and base and not STOP.is_set() and time.time() < A.abort_at:
        last = max(base, key=lambda x: x[2]["train_pass"])  # ties: earliest candidate (content first)
        row["repair"] = []
        messages = list(msgs)
        for i in range(1, A.repair_rounds + 1):
            messages += [{"role": "assistant", "content": "```python\n" + last[1] + "\n```"},
                         {"role": "user", "content": P.feedback(task, last[2])}]
            n2 = prompt_tokens(messages)
            cap_r = min(A.repair_cap, MAX_LEN - n2 - 512)
            if cap_r < 1000 or STOP.is_set() or time.time() >= A.abort_at:
                break
            rr = chat(messages, cap_r, zlib.crc32(f"v13:{task_id}:{rep}:fix{i}".encode()) & 0x7FFFFFFF, effort)
            tokens_total += rr.get("completion_tokens") or 0
            if rr.get("t_end") is not None:
                row["t_end"] = rr["t_end"]
            rcands = [("content", c) for c in code_blocks(rr["content"])[-1:]]
            rcands += [("reasoning", c) for c in code_blocks(rr["reasoning"])[-2:][::-1]]
            n_code = len(code_blocks(rr["content"])) + len(code_blocks(rr["reasoning"]))
            rd = {"round": i, "state": rr.get("state"), "finish_reason": rr.get("finish_reason"),
                  "completion_tokens": rr.get("completion_tokens"), "n_code": n_code, "best_train_pass": None,
                  "verified": False}
            row["repair"].append(rd)
            new = []
            for source, code in rcands:
                if code in seen:
                    continue
                seen.add(code)
                chk = try_code(f"repair{i}:{source}", code)
                new.append((f"repair{i}:{source}", code, chk))
                tried.append(new[-1])
                if row["verified"]:
                    break
            if new:
                rd["best_train_pass"] = max(t[2]["train_pass"] for t in new)
            if row["verified"]:
                rd["verified"] = True
                break
            new = [t for t in new if not t[2].get("rejected")]
            if not new:  # no code, only repeats, or only guard-rejected programs: stop
                break
            last = max(new, key=lambda x: x[2]["train_pass"])
    row["completion_tokens_total"] = tokens_total
    if not row["verified"] and tried:  # the closest program, kept for inspection
        best = max([t for t in tried if not t[2].get("rejected")] or tried, key=lambda x: x[2]["train_pass"])
        row.update(best_code=best[1][:20000], best_train_pass=best[2]["train_pass"], best_train_n=best[2]["train_n"])
    append("v13-programs.jsonl", row)  # the merge reads verified rows from this file
    if row["verified"]:
        with LOCK:
            RESULTS.setdefault(task_id, []).append(row["answers"])
        write_submission()
    return row


def write_submission():
    """Atomic rewrite from the NVARC base + every verified program so far (rerun: submission.json); one lock section,
    so an older snapshot can never land after a newer one."""
    with LOCK:
        merged, changed = M.merge(TEST, BASE, {k: list(v) for k, v in RESULTS.items()})
        M.validate(TEST, merged)
        M.atomic_json(os.path.join(A.work, OUT_NAME), merged)
        S["changed_outputs"] = changed


def job_order():
    """Rerun: NVARC pass-A confidence ascending (no candidate = -inf first), sure tasks (>= CONFIDENT) last; any
    error falls back to file order. Commit: smoke tasks (same rule) then keys30 in list order."""
    keys = list(TEST)
    conf = {}
    try:
        sys.path.insert(0, str(HERE))
        import arc_v6_config as cfg
        conf = cfg.task_confidence(keys, A.outputs, TEST, time.time() + 300)
    except Exception as e:  # noqa: BLE001
        S["order_error"] = f"{type(e).__name__}: {e}"[:300]
    rank = {k: i for i, k in enumerate(keys)}
    if A.commit:
        smoke = sorted((k for k in keys if k in COVERED), key=lambda k: (conf.get(k, float("-inf")), rank[k]))
        order = smoke + [k for k in KEYS30 if k in TEST and k not in smoke]
    else:
        unsure = [k for k in keys if not conf.get(k, float("-inf")) >= CONFIDENT]
        sure = [k for k in keys if conf.get(k, float("-inf")) >= CONFIDENT]
        order = sorted(unsure, key=lambda k: (conf.get(k, float("-inf")), rank[k])) + sure
    S["order"] = {"n": len(order), "first": order[:10], "conf_known": sum(k in conf for k in order),
                  "sure": sum(conf.get(k, float("-inf")) >= CONFIDENT for k in order)}
    return order


def stream_socket(r):
    """The socket under a streaming requests.Response (http.client path; stdlib, stable across urllib3 versions)."""
    for path in (("raw", "_fp", "fp", "raw", "_sock"), ("raw", "_connection", "sock")):
        obj = r
        try:
            for name in path:
                obj = getattr(obj, name)
            if obj is not None:
                return obj
        except AttributeError:
            continue
    return None


def abort_streams():
    """At the abort time: stop the channel and shut down every open stream's socket, so reads blocked on a stream that
    sends nothing (a preempted request) return now; 90 s later the server itself is killed as a last resort."""
    STOP.set()
    with LOCK:
        active = list(ACTIVE.values())
    for r in active:
        try:
            stream_socket(r).shutdown(socket.SHUT_RDWR)
        except Exception:  # noqa: BLE001
            pass
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
    with LOCK:
        S["aborted_streams"] = len(active)
    last = threading.Timer(90, lambda: kill(SERVER[0]))
    last.daemon = True
    last.start()


def phase_jobs(order):
    """Workers pull (task, rep) from a queue that starts as rep 0 of every task in `order`; a job that ends without a
    verified program queues the same task's next rep at the back, so a task is never in flight twice and later reps
    only go to tasks still unsolved."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    queue = deque((k, 0) for k in order)
    cond = threading.Condition()
    inflight = [0]
    done = []

    def worker():
        while True:
            with cond:
                while not queue and inflight[0] and not STOP.is_set() and time.time() < A.stop_new_at:
                    cond.wait(5)
                if not queue or STOP.is_set() or time.time() >= A.stop_new_at:
                    return
                k, rep = queue.popleft()
                inflight[0] += 1
            row = {"verified": False}
            try:
                row = run_job(k, rep)
                done.append(row)
            except Exception as e:  # noqa: BLE001  (one bad job never stops the channel)
                append("v13-programs.jsonl", {"task": k, "rep": rep, "state": "job_exception",
                                              "error": repr(e)[-300:], "verified": False})
            with cond:
                inflight[0] -= 1
                if not row.get("verified") and rep + 1 < MAX_REPS and row.get("state") != "prompt_too_long":
                    queue.append((k, rep + 1))
                cond.notify_all()

    with ThreadPoolExecutor(max_workers=A.conc) as pool:
        for _ in range(A.conc):
            pool.submit(worker)
    return done


def poller():
    keep = ("vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:num_requests_running",
            "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
            "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
            "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")
    while not STOP.wait(30):
        try:
            code, text = http("GET", "/metrics", timeout=10)
            vals = {}
            for line in text.splitlines():
                m = re.match(r"([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
                if m and m.group(1) in keep:
                    vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(3))
            append("v13-metrics.jsonl", {"t": el(), **vals})
        except Exception:  # noqa: BLE001
            pass


def summarize(rows):
    done = [r for r in rows if r.get("state") not in (None, "prompt_too_long")]
    ver = [r for r in rows if r.get("verified")]
    s = {"jobs": len(rows), "finished_stop": sum(r.get("finish_reason") == "stop" for r in done),
         "finished_length": sum(r.get("finish_reason") == "length" for r in done),
         "aborted": sum(r.get("state") == "aborted" for r in done),
         "verified": len(ver), "verified_from_reasoning": sum(str(r.get("source") or "").endswith("reasoning") for r in ver),
         "verified_tasks": len({r["task"] for r in ver}),
         "tokens": sum(r.get("completion_tokens") or 0 for r in rows)}
    s["repaired"] = sum(str(r.get("source") or "").startswith("repair") for r in ver)
    s["repair_rounds_used"] = sum(len(r.get("repair") or []) for r in rows)
    s["tokens_total"] = sum((r.get("completion_tokens_total") if r.get("completion_tokens_total") is not None
                             else r.get("completion_tokens")) or 0 for r in rows)
    if SOLUTIONS:
        s["verified_correct"] = sum(all(r.get("correct") or [False]) for r in ver)
        s["verified_correct_from_reasoning"] = sum(all(r.get("correct") or [False]) for r in ver
                                                   if str(r.get("source") or "").endswith("reasoning"))
    return s


TEST, BASE, TASKS, SOLUTIONS, KEYS30, COVERED, OUT_NAME = {}, {}, {}, {}, [], set(), "submission.json"


def load_tasks():
    global TEST, BASE, TASKS, SOLUTIONS, KEYS30, COVERED, OUT_NAME
    comp = Path(A.comp)
    if A.commit:  # dev measurement on public eval: smoke tasks + keys30, scored against the eval solutions
        TEST = json.loads((comp / "arc-agi_evaluation_challenges.json").read_text())
        sol = comp / "arc-agi_evaluation_solutions.json"
        SOLUTIONS = json.loads(sol.read_text()) if sol.exists() else {}
        KEYS30 = json.loads((HERE / "keys30.json").read_text())
        base_path = Path(A.work) / "eval-smoke-submission.json"
        OUT_NAME = "eval-v13-submission.json"
    else:
        TEST = json.loads((comp / "arc-agi_test_challenges.json").read_text())
        base_path = Path(A.work) / "submission.json"
        OUT_NAME = "submission.json"
    try:
        BASE = json.loads(base_path.read_text())
        if not A.commit:
            M.validate(TEST, BASE)
    except Exception as e:  # noqa: BLE001
        if not A.commit:  # never merge onto an unreadable NVARC submission (placeholders would replace its answers)
            raise Skip(f"NVARC base unreadable/invalid: {type(e).__name__}: {e}"[:300])
        BASE = {}
    TASKS = TEST
    try:
        COVERED = {n.split(".")[0].rsplit("_", 1)[0] for n in os.listdir(A.outputs) if not n.startswith("._tmp_")}
    except OSError:
        COVERED = set()
    if not A.commit:  # the merge base for finalize(); written once, before any program answer
        M.atomic_json(os.path.join(A.work, "v13-base-submission.json"), BASE)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--outputs", default=os.environ.get("ARC_V6_OUTPUTS", "/kaggle/inference_outputs"))
    ap.add_argument("--input-root", default=os.environ.get("ARC_V6_INPUT_ROOT", "/kaggle/input"))
    ap.add_argument("--tmp", default="/tmp/v13")
    ap.add_argument("--comp-dir", default=None)  # the notebook passes cfg.competition_dir()
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--effort-schedule", default=None)  # C2: EFFORT:CAP[,EFFORT:CAP...] per rep (last entry repeats)
    ap.add_argument("--repair-rounds", type=int, default=0)  # C2: execution-feedback repair rounds per job
    ap.add_argument("--repair-cap", type=int, default=8000)
    ap.add_argument("--fake-server", default=None)  # test seams: stub server and shorter margins
    ap.add_argument("--need-seconds", type=float, default=None)
    ap.add_argument("--abort-margin", type=float, default=6 * 60)
    ap.add_argument("--stop-margin", type=float, default=None)
    A = ap.parse_args()
    sched = parse_schedule(A.effort_schedule) if A.effort_schedule is not None else None
    bad = f"effort schedule {A.effort_schedule!r}" if A.effort_schedule is not None and sched is None else None
    if bad is None and not 0 <= A.repair_rounds <= 3:
        bad = f"repair rounds {A.repair_rounds}"
    if bad is None and not 1000 <= A.repair_cap <= 32000:
        bad = f"repair cap {A.repair_cap}"
    if bad is not None:  # validate before any work (tasks, phases, server)
        print(f"bad C2 flag: {bad}", flush=True)
        sys.exit(2)
    A.schedule = sched
    A.commit = not bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    A.cap = A.cap or 64000  # v2: commit = rerun cap (v1 dev run at 20k: 0/66 natural stops; P0b round-1 p50 = 64k)
    if A.schedule:  # the largest cap drives the KV-bound concurrency below
        A.cap = max(c for _, c in A.schedule)
        A.effort = A.schedule[0][0]
    A.abort_at = A.end_time - A.abort_margin  # in-flight streams are cut here; their reasoning drafts are still tried
    A.stop_new_at = A.abort_at - (A.stop_margin if A.stop_margin is not None else (5 if A.commit else 15) * 60)
    S.update(commit=A.commit, cap=A.cap, conc=A.conc, effort=A.effort, end_time=A.end_time,
             seconds_left=round(A.end_time - time.time()), effort_schedule=A.schedule,
             repair_rounds=A.repair_rounds, repair_cap=A.repair_cap)
    proc = None
    stop_poll = None
    try:
        A.comp = A.comp_dir
        if not A.comp:
            comps = [c for c in find("arc-agi_test_challenges.json") if "arc-prize" in c] or \
                find("arc-agi_test_challenges.json")
            A.comp = os.path.dirname(comps[0]) if comps else None
        if not A.comp or not os.path.exists(os.path.join(A.comp, "arc-agi_test_challenges.json")):
            raise Skip("competition files missing")
        load_tasks()
        need = A.need_seconds if A.need_seconds is not None else (25 if A.commit else 45) * 60
        A.setup_deadline = A.stop_new_at - need / 3  # server ready by then, else skip (>= need/3 of job time left)
        if A.stop_new_at - time.time() < need:
            raise Skip(f"not enough time: {round(A.stop_new_at - time.time())}s before the stop")
        wheels, tgz, model = phase_inputs()
        phase_gpus()
        if A.commit and not A.fake_server:
            S["phases"]["model_hash"] = {}
            threading.Thread(target=model_hash, args=(model, S["phases"]["model_hash"]), daemon=True).start()
        vpy = sys.executable if A.fake_server else phase_install(wheels, tgz)
        proc = phase_server(vpy, model, A.setup_deadline)
        SERVER[0] = proc
        if A.schedule:  # an effort the server does not know must skip the phase, not fail every job
            S["effort_smoke"] = effort_smoke([e for e, _ in A.schedule])
        kv = (S["phases"]["server"].get("chosen") or {}).get("kv_tokens")
        if kv:  # KV-bound concurrency (v1 dev run: KV peaked at 0.64 with 22 streams at cap 20k -> 0.35*cap average)
            A.conc = max(8, min(A.conc, int(kv / (0.35 * A.cap + 6000)) + 2))
        S["conc_effective"] = A.conc
        stop_poll = threading.Thread(target=poller, daemon=True)
        stop_poll.start()
        S["sandbox_selftest"] = P.run_program("import numpy as np\ndef transform(g):\n    return np.array(g).T.tolist()\n",
                                              [[[1, 2]]])
        order = job_order()
        S["gen_start"] = el()
        save()
        log(f"jobs start: {len(order)} tasks, conc {A.conc}, cap {A.cap}, stop_new in "
            f"{round(A.stop_new_at - time.time())}s")
        watchdog = threading.Timer(max(0, A.abort_at - time.time()), abort_streams)
        watchdog.daemon = True
        watchdog.start()
        rows = phase_jobs(order)
        S["gen_seconds"] = round(el() - S["gen_start"])
        S["summary"] = summarize(rows)
        log("summary " + json.dumps(S["summary"]))
    except Skip as e:
        S["skipped"] = str(e)
        log(f"skipped: {e}")
    except Exception as e:  # noqa: BLE001
        import traceback
        S["skipped"] = f"error {type(e).__name__}: {e}"[:500]
        S["traceback"] = traceback.format_exc()[-3000:]
        log(f"error: {e}")
    finally:
        STOP.set()
        kill(proc)
        if RESULTS:
            try:
                write_submission()
            except Exception as e:  # noqa: BLE001
                S["final_write_error"] = repr(e)[:300]
        for log_path in glob.glob(os.path.join(A.work, "v13-server-*.log")):  # keep outputs small (P1: 872 MB log)
            try:
                if os.path.getsize(log_path) > 50 << 20:
                    with open(log_path, "rb") as f:
                        f.seek(-(20 << 20), 2)
                        tail = f.read()
                    with open(log_path, "wb") as f:
                        f.write(b"[truncated to last 20 MB]\n" + tail)
            except Exception:  # noqa: BLE001
                pass
        S["elapsed"] = el()
        save()


if __name__ == "__main__":
    main()
