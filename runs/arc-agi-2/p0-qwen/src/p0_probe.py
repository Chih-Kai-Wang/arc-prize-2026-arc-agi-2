"""P0 probe for the Qwen3.8-27B-FP8 line (artifacts/verification/arc-agi-2/new-approach-proposal-v1.md §6).

One Kaggle 4xL4 commit run, offline. Measures, in order, with a hard self-imposed wall clock:
  1. environment, wheel hashes (SHA256SUMS + PyPI pins), model weight sha256 vs the HF LFS list;
  2. vLLM server start (TP=4, FP8 KV, TRITON_ATTN pinned, MTP; fallback ladder) and KV capacity from its log;
  3. long-reasoning throughput under load via /metrics, and per-request tokens, truncation, parse rate and accuracy
     for reasoning_effort xhigh x1 and low x2 on colour/dihedral-augmented ARC-AGI-2 public-eval tasks (keys30);
  4. if time remains, KV capacity and short-context throughput of a TP=2 replica.
Everything is written incrementally to $P0_OUT (summary.json, requests.jsonl, metrics.jsonl, probe.log, server logs).
Never touches the network (pip --no-index, HF offline). Exits 0 on every outcome so the outputs are downloadable."""
import glob
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arc_llm as L  # noqa: E402

T0 = time.time()
WALL = float(os.environ.get("P0_WALL_SECONDS", 3 * 3600 + 10 * 60))
OUT = Path(os.environ.get("P0_OUT", "/kaggle/working/p0"))
INPUT = Path(os.environ.get("P0_INPUT", "/kaggle/input"))
HERE = Path(__file__).resolve().parent
N_TASKS = int(os.environ.get("P0_TASKS", 20))
CAPS = {"xhigh": 120000, "low": 80000}
MAX_LEN = 131072
MAIN_STOP = T0 + WALL - 35 * 60   # no new main-phase requests after this
MAIN_ABORT = T0 + WALL - 28 * 60  # in-flight main-phase requests are cut here (recorded as censored)
TAIL_NEED = 20 * 60               # TP=2 tail only if this much remains before the hard end
END_RESERVE = 6 * 60              # always keep this for writing outputs
PYPI = {  # sha256 of the official PyPI wheels (pypi.org JSON API, 2026-10-12)
    "vllm-0.19.0-cp38-abi3-manylinux_2_31_x86_64.whl":
        "2d0e5fae45367bdbf111fcad68f4c0f8fdddd2f2fb643e52f0f2daebef7b41cf",
    "torch-2.10.0-3-cp312-cp312-manylinux_2_28_x86_64.whl":
        "98c01b8bb5e3240426dcde1446eed6f40c778091c8544767ef1168fc663a05a6",
}
KEYS30 = json.loads((HERE / "keys30.json").read_text())  # E1 keys30 (gcp-e1/make_inputs.py), public eval

HASH_POOL = []
ACTIVE = {}  # id(row) -> streaming response, closed by the watchdog at abort time
S = {"version": "p0-v1", "t0": T0, "wall_seconds": WALL, "phases": {}, "fatal": None}
_lock = threading.Lock()
ABORT = threading.Event()


def el():
    return round(time.time() - T0, 1)


def log(msg):
    line = f"[{el():8.1f}s] {msg}"
    print(line, flush=True)
    with _lock, (OUT / "probe.log").open("a") as f:
        f.write(line + "\n")


def save():
    with _lock:
        tmp = OUT / "summary.json.tmp"
        tmp.write_text(json.dumps(S, indent=1, default=str))
        tmp.replace(OUT / "summary.json")


def append(name, row):
    with _lock, (OUT / name).open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def run(cmd, timeout=None, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_dir(pattern):
    hits = sorted(glob.glob(str(INPUT / "**" / pattern), recursive=True))
    return Path(hits[0]).parent if hits else None


# --- 1. environment and integrity ------------------------------------------------------------------------------------

def phase_env():
    ph = S["phases"]["env"] = {"python": sys.version}
    for name, cmd in [("nvidia_smi", ["nvidia-smi"]), ("df", ["df", "-h", "/kaggle/working", "/tmp"]),
                      ("nproc", ["nproc"]), ("free", ["free", "-g"])]:
        try:
            ph[name] = run(cmd, timeout=30)[1][-3000:]
        except Exception as e:  # noqa: BLE001
            ph[name] = repr(e)
    comp, model, wheels = (find_dir("arc-agi_evaluation_challenges.json"), find_dir("layers-0.safetensors"),
                           find_dir("vllm-0.19.0-*.whl"))
    ph.update(comp_dir=str(comp), model_dir=str(model), wheel_dir=str(wheels))
    log(f"python {sys.version.split()[0]} comp={comp} model={model} wheels={wheels}")
    save()
    if not (comp and model and wheels):
        raise SystemExit("input mount missing")
    return comp, model, wheels


def verify_wheels(wheels):
    ph = S["phases"]["wheels"] = {}
    sums = {}
    for line in (wheels / "SHA256SUMS").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]
    files = sorted(p.name for p in wheels.glob("*.whl"))
    bad = [n for n in files if sums.get(n) != sha_file(wheels / n)]
    pypi_ok = {n: (sums.get(n) == h and (wheels / n).exists()) for n, h in PYPI.items()}
    ph.update(n_wheels=len(files), mismatched=bad, unlisted=[n for n in files if n not in sums], pypi_pins=pypi_ok)
    log(f"wheels {len(files)} mismatched={len(bad)} pypi_pins={pypi_ok}")
    save()
    if bad or not all(pypi_ok.values()):
        raise SystemExit("wheel hash check failed")


def start_model_hash(model):
    ref = json.loads((HERE / "qwen38_fp8_sha256.json").read_text())["files"]
    pool = ThreadPoolExecutor(max_workers=8)
    futs = {n: pool.submit(sha_file, model / n) for n in ref if (model / n).exists()}
    return ref, futs, pool


def finish_model_hash(model, ref, futs, pool):
    ph = S["phases"]["model_hash"] = {}
    missing = [n for n in ref if n not in futs]
    bad = [n for n, f in futs.items() if f.result() != ref[n][1]]
    pool.shutdown()
    sizes_bad = [n for n in futs if (model / n).stat().st_size != ref[n][0]]
    ph.update(files=len(ref), missing=missing, mismatched=bad, size_mismatched=sizes_bad)
    log(f"model sha256: {len(ref)} files, missing={missing} mismatched={bad}")
    save()
    if missing or bad or sizes_bad:
        raise SystemExit("model hash check failed")


def install(wheels):
    ph = S["phases"]["install"] = {}
    t = time.time()
    rc, out = run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-index", "--find-links",
                   str(wheels), "--requirement", str(wheels / "requirements.lock")], timeout=1800)
    ph.update(rc=rc, seconds=round(time.time() - t), tail=out[-2500:])
    log(f"pip install rc={rc} in {ph['seconds']}s")
    save()
    if rc != 0:
        raise SystemExit("pip install failed")
    rc, out = run([sys.executable, "-c", "import json, torch, vllm, transformers; print(json.dumps({'torch': "
                   "torch.__version__, 'cuda': torch.version.cuda, 'cuda_ok': torch.cuda.is_available(), "
                   "'gpus': torch.cuda.device_count(), 'vllm': vllm.__version__, 'transformers': "
                   "transformers.__version__}))"], timeout=600)
    ph["import"] = out.strip().splitlines()[-1] if out.strip() else out
    log(f"import check rc={rc}: {ph['import']}")
    rc2, _ = run([sys.executable, "-c", "import flash_attn"], timeout=120)
    ph["flash_attn"] = rc2 == 0
    if rc2 != 0:  # vLLM forwards layer_idx to an attention impl that rejects it when flash-attn is absent (public
        # Apache-2.0 notebook delongyang/arc-agi-2-inference-qwen3-8-27b-fp8-within-4-hours documents this fix)
        rc3, root = run([sys.executable, "-c", "import os, vllm; print(os.path.dirname(vllm.__file__))"], timeout=300)
        root = root.strip().splitlines()[-1] if rc3 == 0 and root.strip() else ""
        patched, reverted = [], []
        for f in sorted(glob.glob(root + "/**/attention/*.py", recursive=True)) if root else []:
            text = Path(f).read_text()
            if "self.impl = impl_cls(" not in text or "extra_impl_args.pop('layer_idx', None)" in text:
                continue
            new = re.sub(r"^(\s*)self\.impl = impl_cls\(", r"\1extra_impl_args.pop('layer_idx', None)\n\1self.impl = impl_cls(",
                         text, count=1, flags=re.M)
            Path(f + ".bak").write_text(text)
            Path(f).write_text(new)
            if run([sys.executable, "-m", "py_compile", f], timeout=120)[0] != 0 or "extra_impl_args" not in text:
                Path(f).write_text(text)  # revert: does not compile, or the variable does not exist in this version
                reverted.append(f)
            else:
                patched.append(f)
        ph.update(attention_patch=patched, attention_patch_reverted=reverted, vllm_root=root)
        log(f"attention patch: patched={patched} reverted={reverted}")
    save()
    if rc != 0 or '"cuda_ok": true' not in ph["import"]:
        raise SystemExit("torch/vllm import or CUDA check failed")


# --- 2. vLLM server ------------------------------------------------------------------------------------------------------

def server_env(gpus=None, force_backend=True):
    env = dict(os.environ, PYTORCH_ALLOC_CONF="expandable_segments:True",
               VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", VLLM_LOGGING_LEVEL="INFO")
    libcuda = "/usr/local/nvidia/lib64/libcuda.so"
    if os.path.exists(libcuda):  # same libcuda pin as the public notebook above
        env["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + env.get("LD_LIBRARY_PATH", "")
        env["LD_PRELOAD"] = libcuda
    if force_backend:  # pins the MTP drafter too (public notebook above, vLLM PR #53450)
        env["VLLM_FORCE_ATTN_BACKEND"] = "TRITON_ATTN"
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    return env


SUPPORTED = None


def supported_flags():
    global SUPPORTED
    if SUPPORTED is None:
        rc, out = run([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help=all"], timeout=600)
        SUPPORTED = set(re.findall(r"(--[a-z0-9][a-z0-9-]+)", out)) if rc == 0 else set()
        S["phases"]["server_flags"] = {"rc": rc, "n": len(SUPPORTED)}
    return SUPPORTED


def start_server(model, port, tp, mtp, mm_off, gpus=None, tag="main", safe=False):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
           "--served-model-name", "q38", "--host", "127.0.0.1", "--port", str(port),
           "--tensor-parallel-size", str(tp), "--gpu-memory-utilization", "0.88" if safe else "0.93",
           "--max-model-len", str(MAX_LEN), "--max-num-seqs", "48", "--reasoning-parser", "qwen3",
           "--trust-remote-code"]
    if safe:  # last rung: default KV dtype and backend choice, eager mode, lower memory target
        cmd += ["--enforce-eager"]
    else:
        cmd += ["--kv-cache-dtype", "fp8", "--attention-backend", "TRITON_ATTN", "--no-async-scheduling"]
    if mm_off:
        cmd += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if mtp:
        cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": mtp})]
    known = supported_flags()
    if known:  # drop flags this vLLM build does not list (logged), rather than fail every rung on one bad flag
        dropped, kept, i = [], [], 0
        while i < len(cmd):
            a = cmd[i]
            takes = i + 1 < len(cmd) and not cmd[i + 1].startswith("--")
            if a.startswith("--") and a not in known:
                dropped.append(a)
                i += 2 if takes and a not in ("--trust-remote-code", "--enforce-eager", "--no-async-scheduling") else 1
                continue
            kept.append(a)
            i += 1
        cmd = kept
        if dropped:
            log(f"dropped unsupported flags: {dropped}")
    path = OUT / f"server-{tag}-tp{tp}-mtp{mtp}-mm{int(mm_off)}{'-safe' if safe else ''}.log"
    fh = path.open("w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                            start_new_session=True, env=server_env(gpus, force_backend=not safe))
    return proc, path, cmd


def http(method, url, body=None, timeout=30):
    import requests
    r = requests.request(method, url, json=body, timeout=timeout)
    return r.status_code, r.text


def wait_ready(proc, port, deadline):
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if http("GET", f"http://127.0.0.1:{port}/v1/models", timeout=5)[0] == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def smoke(port):
    body = {"model": "q38", "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    code, text = http("POST", f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout=300)
    try:
        msg = json.loads(text)["choices"][0]["message"]
    except Exception:  # noqa: BLE001
        return False, text[-600:]
    txt = (msg.get("content") or "") + (msg.get("reasoning_content") or msg.get("reasoning") or "")
    return code == 200 and len(txt.strip()) > 0, text[-600:]


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
    for _ in range(30):  # wait for VRAM release
        rc, out = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=30)
        if rc == 0 and all(int(x) < 1500 for x in out.split() if x.strip().isdigit()):
            return
        time.sleep(4)


def kv_capacity(path):
    text = path.read_text(errors="ignore") if path.exists() else ""
    tok = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    conc = re.findall(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", text)
    backend = sorted(set(re.findall(r"[Uu]sing (?:AttentionBackendEnum\.)?([A-Z][A-Z_]+)(?: attention)? backend", text)))
    return {"kv_tokens": int(tok[-1].replace(",", "")) if tok else None,
            "max_concurrency_at_len": float(conc[-1][1]) if conc else None, "backends_logged": backend}


def phase_server(model):
    ph = S["phases"]["server"] = {"attempts": []}
    for mtp, mm_off, safe in [(2, True, False), (0, True, False), (0, False, False), (0, True, True)]:
        if time.time() > T0 + WALL * 0.45:
            break
        proc, path, cmd = start_server(model, 8000, 4, mtp, mm_off, safe=safe)
        t = time.time()
        ready = wait_ready(proc, 8000, min(t + 30 * 60, T0 + WALL * 0.5))
        att = {"mtp": mtp, "mm_off": mm_off, "safe": safe, "ready": ready, "startup_seconds": round(time.time() - t),
               "log": path.name, "cmd": cmd}
        if ready:
            ok, resp = smoke(8000)
            att.update(smoke_ok=ok, smoke_resp=resp, **kv_capacity(path))
            if ok:
                ph["attempts"].append(att)
                ph["chosen"] = att
                log(f"server ready: mtp={mtp} mm_off={mm_off} kv={att['kv_tokens']} "
                    f"conc@131k={att['max_concurrency_at_len']} backends={att['backends_logged']}")
                save()
                return proc, att
        att["log_tail"] = path.read_text(errors="ignore")[-3000:] if path.exists() else ""
        ph["attempts"].append(att)
        log(f"server attempt failed: mtp={mtp} mm_off={mm_off} ready={ready}")
        save()
        kill(proc)
    raise SystemExit("no server configuration started")


# --- 3. metrics + requests -------------------------------------------------------------------------------------------

METRICS = ("vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:num_requests_running",
           "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
           "vllm:num_preemptions_total", "vllm:spec_decode_num_accepted_tokens_total",
           "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_drafts_total")


def scrape(port):
    vals = {}
    try:
        code, text = http("GET", f"http://127.0.0.1:{port}/metrics", timeout=10)
    except Exception:  # noqa: BLE001
        return None
    if code != 200:
        return None
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m and m.group(1) in METRICS:
            vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(3))
    return vals


def poller(port, stop, tag):
    while not stop.is_set():
        v = scrape(port)
        if v is not None:
            append("metrics.jsonl", {"t": el(), "tag": tag, **v})
        stop.wait(15)


def prompt_tokens(port, messages, effort):
    """Token count of the templated prompt; retries without template kwargs, then falls back to the character count
    (an upper bound for these digit/comma grids) so max_tokens never overflows the context."""
    for body in ({"model": "q38", "messages": messages,
                  "chat_template_kwargs": {"reasoning_effort": effort, "enable_thinking": True}},
                 {"model": "q38", "messages": messages}):
        try:
            code, text = http("POST", f"http://127.0.0.1:{port}/tokenize", body, timeout=30)
            if code == 200 and json.loads(text).get("count"):
                return json.loads(text)["count"], "tokenize"
        except Exception:  # noqa: BLE001
            pass
    return sum(len(m["content"]) for m in messages) + 300, "chars"


def one_request(job, port, abort_at, stop_at=None):
    import requests
    row = {k: job[k] for k in ("key", "task", "test_index", "effort", "rep", "aug_id", "phase")}
    row.update(state="not_started", t_queue=el())
    if time.time() > (MAIN_STOP if stop_at is None else stop_at) or ABORT.is_set():
        append("requests.jsonl", row)
        return row
    messages = [{"role": "user", "content": job["prompt"]}]
    n_prompt, how = prompt_tokens(port, messages, job["effort"])
    cap = min(CAPS[job["effort"]], MAX_LEN - n_prompt - 64)
    if cap < 4096:
        row.update(state="prompt_too_long", prompt_tokens_est=n_prompt)
        append("requests.jsonl", row)
        return row
    body = {"model": "q38", "messages": messages, "max_tokens": cap, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "seed": job["seed"], "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"reasoning_effort": job["effort"], "enable_thinking": True}}
    row.update(state="running", t_start=el(), max_tokens=cap, prompt_tokens_est=n_prompt, prompt_count_method=how)
    content, reasoning_chars, head, tail = [], 0, "", ""
    usage, finish = {}, None
    try:
        with requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, stream=True,
                           timeout=(30, 600)) as r:
            ACTIVE[id(row)] = r
            if r.status_code != 200:
                row.update(state="http_error", error=r.text[-800:])
                return row
            for raw in r.iter_lines(decode_unicode=True):
                if time.time() > abort_at or ABORT.is_set():
                    row["state"] = "aborted"
                    break
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[6:]
                if data.strip() == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    rc = d.get("reasoning_content") or d.get("reasoning") or ""
                    if rc:
                        if len(head) < 1500:
                            head += rc[:1500 - len(head)]
                        tail = (tail + rc)[-1500:]
                        reasoning_chars += len(rc)
                    if d.get("content"):
                        content.append(d["content"])
                        if "t_first_content" not in row:
                            row["t_first_content"] = el()
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001
        stopped = time.time() > abort_at or ABORT.is_set()
        row.update(state="aborted" if stopped else "exception", error=repr(e)[-800:])
    finally:
        ACTIVE.pop(id(row), None)
    text = "".join(content)
    row.update(t_end=el(), prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
               finish_reason=finish, reasoning_chars=reasoning_chars, reasoning_head=head, reasoning_tail=tail,
               content_tail=text[-2000:])
    if row["state"] == "running":
        row["state"] = "done" if finish else "incomplete"
    ans = L.parse_answer(text) if row["state"] == "done" else None
    if ans is not None:
        ans = L.invert_aug(ans, job["aug"])
    row["answer"] = ans
    row["correct"] = ans is not None and ans == job["solution"]
    append("requests.jsonl", row)
    return row


def build_jobs(comp, phase="main"):
    """Per test output: xhigh and low rep0 on augmentation A, low rep1 on augmentation B (agreement across different
    augmentations is the stronger confidence signal); answers are inverted to the original frame before scoring."""
    ch = json.loads((comp / "arc-agi_evaluation_challenges.json").read_text())
    so = json.loads((comp / "arc-agi_evaluation_solutions.json").read_text())
    jobs, keys = [], [k for k in KEYS30 if k in ch][:N_TASKS]
    for k in keys:
        augs = {"A": L.augmentation(k), "B": L.augmentation(k, salt="p0-v1-B")}
        tasks = {a: L.augment_task(ch[k], g) for a, g in augs.items()}
        for i in range(len(ch[k]["test"])):
            for effort, rep, a in (("xhigh", 0, "A"), ("low", 0, "A"), ("low", 1, "B")):
                jobs.append({"key": f"{k}:{i}", "task": k, "test_index": i, "effort": effort, "rep": rep,
                             "aug_id": a, "phase": phase,
                             "seed": zlib.crc32(f"{k}:{i}:{effort}:{rep}".encode()) & 0x7FFFFFFF,
                             "prompt": L.build_prompt(tasks[a], i), "aug": augs[a], "solution": so[k][i]})
    return keys, jobs


def effort_check(port, jobs):
    """Does reasoning_effort reach the template? Token counts of the same prompt at xhigh vs low must differ."""
    msgs = [{"role": "user", "content": jobs[0]["prompt"]}]
    counts = {e: prompt_tokens(port, msgs, e) for e in ("xhigh", "low")}
    return {"counts": counts, "differ": counts["xhigh"] != counts["low"]}


def watchdog(proc, stop, abort_at):
    """Close live streams at abort time (a stalled stream otherwise blocks its worker up to the read timeout) and
    record a server death, which ends the phase early instead of failing every queued job slowly."""
    while not stop.is_set():
        if time.time() > abort_at:
            ABORT.set()
        if proc is not None and proc.poll() is not None:
            S["phases"].setdefault("main", {})["server_died_at"] = el()
            log("server process died")
            ABORT.set()
        if ABORT.is_set():
            for r in list(ACTIVE.values()):
                try:
                    r.close()
                except Exception:  # noqa: BLE001
                    pass
        stop.wait(5)


def phase_main(comp, att, port=8000, proc=None):
    ph = S["phases"]["main"] = {}
    keys, jobs = build_jobs(comp)
    y = att.get("max_concurrency_at_len") or 6.0
    conc = int(max(4, min(48, round(y * 1.8))))
    ph.update(tasks=keys, outputs=len({j["key"] for j in jobs}), jobs=len(jobs), concurrency=conc,
              t_start=el(), stop_at=round(MAIN_STOP - T0), abort_at=round(MAIN_ABORT - T0))
    ph["effort_check"] = effort_check(port, jobs)
    log(f"main: {len(keys)} tasks, {ph['outputs']} outputs, {len(jobs)} jobs, concurrency {conc}, "
        f"effort_check {ph['effort_check']}")
    save()
    stop = threading.Event()
    th = threading.Thread(target=poller, args=(port, stop, "main"), daemon=True)
    wd = threading.Thread(target=watchdog, args=(proc, stop, MAIN_ABORT), daemon=True)
    th.start()
    wd.start()
    rows = []
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(one_request, j, port, MAIN_ABORT) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
            if len(rows) % 10 == 0:
                ph["progress"] = {"rows": len(rows), "t": el()}
                save()
    stop.set()
    th.join(timeout=30)
    wd.join(timeout=30)
    ABORT.clear()
    ph["t_end"] = el()
    ph["arms"] = {e: L.arm_summary([r for r in rows if r["effort"] == e]) for e in ("xhigh", "low")}
    ph["low_pair_agreement"] = L.pair_agreement([r for r in rows if r["effort"] == "low"])
    by_key = {}
    for r in rows:
        by_key.setdefault(r["key"], []).append(r)
    ph["union_any_correct_outputs"] = sum(any(x.get("correct") for x in v) for v in by_key.values())
    ph["throughput"] = throughput("main", ph["t_start"], ph["t_end"], conc)
    log(f"main done: arms={json.dumps(ph['arms'])} throughput={json.dumps(ph['throughput'])}")
    save()


def throughput(tag, t_start, t_end, conc=None):
    p = OUT / "metrics.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    rows = [r for r in rows if r.get("tag") == tag and "vllm:generation_tokens_total" in r]
    if len(rows) < 2:
        return {"samples": len(rows)}
    g = "vllm:generation_tokens_total"
    pairs = [(a, b) for a, b in zip(rows, rows[1:]) if b["t"] > a["t"]]
    rate = lambda ps: (sum(b[g] - a[g] for a, b in ps) / sum(b["t"] - a["t"] for a, b in ps)) if ps else None
    loaded = [(a, b) for a, b in pairs if conc and a.get("vllm:num_requests_running", 0) >= 0.8 * conc
              and a["t"] >= t_start + 300]
    per_req = [(b[g] - a[g]) / (b["t"] - a["t"]) / a["vllm:num_requests_running"] for a, b in loaded
               if a.get("vllm:num_requests_running")]
    a, b = rows[0], rows[-1]
    out = {"samples": len(rows), "overall_tok_per_s": round(rate(pairs), 1),
           "loaded_tok_per_s": round(rate(loaded), 1) if loaded else None, "loaded_seconds": round(sum(
               y["t"] - x["t"] for x, y in loaded)),
           "per_running_request_tok_per_s": round(sum(per_req) / len(per_req), 2) if per_req else None,
           "running_mean": round(sum(r.get("vllm:num_requests_running", 0) for r in rows) / len(rows), 2),
           "kv_usage_max": max(max(r.get("vllm:kv_cache_usage_perc", 0), r.get("vllm:gpu_cache_usage_perc", 0))
                               for r in rows),
           "preemptions": b.get("vllm:num_preemptions_total", 0) - a.get("vllm:num_preemptions_total", 0)}
    acc = b.get("vllm:spec_decode_num_accepted_tokens_total", 0) - a.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    draft = b.get("vllm:spec_decode_num_draft_tokens_total", 0) - a.get("vllm:spec_decode_num_draft_tokens_total", 0)
    out["spec_acceptance"] = round(acc / draft, 3) if draft else None
    return out


# --- 4. TP=2 tail ------------------------------------------------------------------------------------------------------

def phase_tp2(model, comp, att):
    ph = S["phases"]["tp2"] = {}
    proc, path, _ = start_server(model, 8001, 2, att["mtp"], att["mm_off"], gpus="0,1", tag="tail")
    deadline = T0 + WALL - END_RESERVE - 5 * 60
    ready = wait_ready(proc, 8001, deadline)
    ph.update(ready=ready, **kv_capacity(path))
    if ready and time.time() < deadline:
        _, jobs = build_jobs(comp, phase="tail")
        xs = [j for j in jobs if j["effort"] == "xhigh"]
        conc = int(max(2, min(24, round((ph.get("max_concurrency_at_len") or 3) * 1.8))))
        stop = threading.Event()
        end = min(time.time() + 240, deadline)
        th = threading.Thread(target=poller, args=(8001, stop, "tp2"), daemon=True)
        wd = threading.Thread(target=watchdog, args=(proc, stop, end), daemon=True)
        t_start = el()
        th.start()
        wd.start()
        with ThreadPoolExecutor(max_workers=conc) as pool:
            list(pool.map(lambda j: one_request(j, 8001, end, stop_at=end), xs[:conc]))
        stop.set()
        th.join(timeout=30)
        wd.join(timeout=30)
        rows = [json.loads(x) for x in (OUT / "metrics.jsonl").read_text().splitlines() if '"tp2"' in x]
        if len(rows) >= 2:
            dt = max(1.0, rows[-1]["t"] - rows[0]["t"])
            ph["short_context_tok_per_s"] = round((rows[-1].get("vllm:generation_tokens_total", 0)
                                                   - rows[0].get("vllm:generation_tokens_total", 0)) / dt, 1)
        ph.update(concurrency=conc, t_start=t_start)
    log(f"tp2 tail: {json.dumps(ph)}")
    save()
    kill(proc)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        comp, model, wheels = phase_env()
        ref, futs, pool = start_model_hash(model)
        HASH_POOL.append(pool)
        verify_wheels(wheels)
        install(wheels)
        finish_model_hash(model, ref, futs, pool)
        proc, att = phase_server(model)
        phase_main(comp, att, proc=proc)
        kill(proc)
        proc = None
        if T0 + WALL - time.time() > TAIL_NEED:
            phase_tp2(model, comp, att)
        else:
            S["phases"]["tp2"] = {"skipped": "not enough time"}
    except SystemExit as e:
        S["fatal"] = str(e)
        log(f"FATAL: {e}")
    except Exception as e:  # noqa: BLE001
        import traceback
        S["fatal"] = repr(e)
        S["traceback"] = traceback.format_exc()[-4000:]
        log(f"EXCEPTION: {e!r}")
    finally:
        ABORT.set()
        for pool_ in HASH_POOL:
            pool_.shutdown(wait=False, cancel_futures=True)
        kill(proc)
        S["elapsed_seconds"] = el()
        save()
        log("probe finished")


if __name__ == "__main__":
    main()
