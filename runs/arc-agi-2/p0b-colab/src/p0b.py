"""P0b (Colab H100): program-synthesis channel vs direct answers for one model on the keys30 public-eval tasks
(augmented). Per task two independent program samples (augmentation A and B), each verified on the train pairs and
repaired once with execution feedback; then, with leftover time, one direct xhigh answer per test output.
Measures tokens per call, verification rate, verified->test precision, and outputs solved; streams to --out.
  python p0b.py --model-dir /content/m38 --tag q38 --minutes 40 --out /content/out --data /content/data"""
import argparse
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
import progsynth as P  # noqa: E402

A = None
T0 = time.time()
LOCK = threading.Lock()
MAX_LEN = 131072


def el():
    return round(time.time() - T0, 1)


def log(msg):
    line = f"[{A.tag} {el():7.1f}s] {msg}"
    print(line, flush=True)
    with LOCK, (A.out / f"{A.tag}.log").open("a") as f:
        f.write(line + "\n")


def append(row):
    with LOCK, (A.out / f"{A.tag}-calls.jsonl").open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def http(method, path, body=None, timeout=30):
    import requests
    r = requests.request(method, f"http://127.0.0.1:{A.port}{path}", json=body, timeout=timeout)
    return r.status_code, r.text


def start_server():
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(A.model_dir),
           "--served-model-name", "m", "--host", "127.0.0.1", "--port", str(A.port), "--tensor-parallel-size", "1",
           "--gpu-memory-utilization", "0.92", "--max-model-len", str(MAX_LEN), "--max-num-seqs", "64",
           "--kv-cache-dtype", "fp8", "--reasoning-parser", "qwen3", "--trust-remote-code",
           "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if A.mtp:
        cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": A.mtp})]
    fh = (A.out / f"{A.tag}-server.log").open("w")
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                            start_new_session=True, env=dict(os.environ, VLLM_NO_USAGE_STATS="1"))


def wait_ready(proc, limit):
    end = time.time() + limit
    while time.time() < end and proc.poll() is None:
        try:
            if http("GET", "/v1/models", timeout=5)[0] == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def chat(messages, cap, effort, seed, abort_at):
    import requests
    body = {"model": "m", "messages": messages, "max_tokens": cap, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"reasoning_effort": effort, "enable_thinking": True}}
    out = {"state": "running", "t_start": el(), "max_tokens": cap}
    content, usage, finish = [], {}, None
    try:
        with requests.post(f"http://127.0.0.1:{A.port}/v1/chat/completions", json=body, stream=True,
                           timeout=(30, 600)) as r:
            if r.status_code != 200:
                return {**out, "state": "http_error", "error": r.text[-500:]}
            for raw in r.iter_lines(decode_unicode=True):
                if time.time() > abort_at:
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
                    finish = ch.get("finish_reason") or finish
    except Exception as e:  # noqa: BLE001
        out.update(state="exception", error=repr(e)[-500:])
    out.update(t_end=el(), completion_tokens=usage.get("completion_tokens"), prompt_tokens=usage.get("prompt_tokens"),
               finish_reason=finish, content="".join(content))
    if out["state"] == "running":
        out["state"] = "done" if finish else "incomplete"
    return out


def program_job(task_id, raw, rep, abort_at):
    aug = L.augmentation(task_id, salt="p0-v1" if rep == 0 else "p0-v1-B")
    task = L.augment_task(raw["challenge"], aug)
    seed = zlib.crc32(f"prog:{task_id}:{rep}".encode()) & 0x7FFFFFFF
    msgs = [{"role": "user", "content": P.build_prompt(task)}]
    row = {"kind": "program", "task": task_id, "rep": rep, "rounds": []}
    for rnd in range(1 + A.repairs):
        if time.time() > abort_at:
            break
        r = chat(msgs, A.cap_program if rnd == 0 else A.cap_repair, A.effort, seed + rnd, abort_at)
        code = P.extract_code(r["content"]) if r["state"] == "done" else None
        chk = P.check(code, task) if code else {"verified": False, "error": "no code", "train_pass": 0,
                                                 "train_n": len(task["train"]), "tests": None}
        info = {k: r.get(k) for k in ("state", "completion_tokens", "prompt_tokens", "finish_reason", "t_start",
                                      "t_end")}
        info.update(has_code=code is not None, verified=chk["verified"], train_pass=chk["train_pass"],
                    train_n=chk["train_n"], error=chk.get("error"), code_tail=(code or "")[-1500:])
        row["rounds"].append(info)
        if chk["verified"]:
            answers = [L.invert_aug(t, aug) for t in chk["tests"]]
            row.update(verified=True, round_verified=rnd, answers=answers,
                       correct=[a == s for a, s in zip(answers, raw["solution"])])
            break
        if r["state"] != "done" or rnd == A.repairs:
            break
        msgs = msgs + [{"role": "assistant", "content": r["content"][-6000:]},
                       {"role": "user", "content": P.feedback(task, chk)}]
    row.setdefault("verified", False)
    row["tokens"] = sum((x.get("completion_tokens") or 0) for x in row["rounds"])
    append(row)
    return row


def direct_job(task_id, raw, i, abort_at):
    aug = L.augmentation(task_id)
    task = L.augment_task(raw["challenge"], aug)
    r = chat([{"role": "user", "content": L.build_prompt(task, i)}], A.cap_direct, A.effort,
             zlib.crc32(f"direct:{task_id}:{i}".encode()) & 0x7FFFFFFF, abort_at)
    ans = L.parse_answer(r["content"]) if r["state"] == "done" else None
    ans = L.invert_aug(ans, aug) if ans is not None else None
    row = {"kind": "direct", "task": task_id, "test_index": i, "answer": ans, "correct": ans == raw["solution"][i],
           **{k: r.get(k) for k in ("state", "completion_tokens", "prompt_tokens", "finish_reason", "t_start", "t_end")},
           "tokens": r.get("completion_tokens") or 0}
    append(row)
    return row


def poller(stop):
    while not stop.is_set():
        try:
            code, text = http("GET", "/metrics", timeout=10)
            vals = {}
            for line in text.splitlines():
                m = re.match(r"(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
                if m and m.group(1) in ("vllm:generation_tokens_total", "vllm:num_requests_running",
                                        "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
                                        "vllm:spec_decode_num_accepted_tokens_total",
                                        "vllm:spec_decode_num_draft_tokens_total"):
                    vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(3))
            with LOCK, (A.out / f"{A.tag}-metrics.jsonl").open("a") as f:
                f.write(json.dumps({"t": el(), **vals}) + "\n")
        except Exception:  # noqa: BLE001
            pass
        stop.wait(15)


def summarize(prog, direct):
    by = {}
    for r in prog:
        by.setdefault(r["task"], []).append(r)
    ver_tasks = sum(any(x["verified"] for x in v) for v in by.values())
    ver_progs = [r for r in prog if r["verified"]]
    prec = sum(all(r["correct"]) for r in ver_progs)
    solved_prog = 0
    n_outputs = 0
    for t, v in by.items():
        sol_n = len(DATA[t]["solution"])
        n_outputs += sol_n
        ok = [False] * sol_n
        for r in v:  # pass@2 over the two samples, verified-first
            if r["verified"]:
                ok = [a or b for a, b in zip(ok, r["correct"])]
        solved_prog += sum(ok)
    tok_r1 = [x["rounds"][0]["completion_tokens"] for x in prog if x["rounds"] and x["rounds"][0]["state"] == "done"]
    tok_job = [x["tokens"] for x in prog]
    dir_done = [r for r in direct if r["state"] == "done"]
    return {"program": {"jobs": len(prog), "tasks": len(by), "verified_tasks": ver_tasks,
                        "verified_programs": len(ver_progs), "verified_all_tests_correct": prec,
                        "precision": round(prec / len(ver_progs), 3) if ver_progs else None,
                        "verified_in_round": [sum(1 for r in ver_progs if r["round_verified"] == k)
                                              for k in range(1 + A.repairs)],
                        "outputs_solved_pass2": solved_prog, "outputs_in_tasks": n_outputs,
                        "tokens_round1": L.quantiles(tok_r1) | {"mean": round(sum(tok_r1) / len(tok_r1)) if tok_r1 else None,
                                                                "n": len(tok_r1)},
                        "tokens_per_job_mean": round(sum(tok_job) / len(tok_job)) if tok_job else None},
            "direct": {"calls": len(direct), "done": len(dir_done), "correct": sum(r["correct"] for r in dir_done),
                       "tokens": L.token_stats(direct)}}


DATA = {}


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--out", type=Path, default=Path("/content/out"))
    ap.add_argument("--data", type=Path, default=Path("/content/data"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mtp", type=int, default=2)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--repairs", type=int, default=1)
    ap.add_argument("--cap-program", type=int, default=64000)
    ap.add_argument("--cap-repair", type=int, default=32000)
    ap.add_argument("--cap-direct", type=int, default=100000)
    A = ap.parse_args()
    A.out.mkdir(parents=True, exist_ok=True)
    keys = json.loads((Path(__file__).resolve().parent / "keys30.json").read_text())
    ch = json.loads((A.data / "arc-agi_evaluation_challenges.json").read_text())
    so = json.loads((A.data / "arc-agi_evaluation_solutions.json").read_text())
    for k in keys:
        DATA[k] = {"challenge": ch[k], "solution": so[k]}
    summary = {"tag": A.tag, "model_dir": str(A.model_dir), "args": {k: str(v) for k, v in vars(A).items()}}
    proc = start_server()
    t = time.time()
    ready = wait_ready(proc, 1800)
    summary.update(ready=ready, startup_seconds=round(time.time() - t))
    log(f"server ready={ready} in {summary['startup_seconds']}s")
    if not ready and A.mtp:  # retry once without MTP
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(20)
        A.mtp = 0
        proc = start_server()
        ready = wait_ready(proc, 1500)
        summary.update(ready_no_mtp=ready)
    prog, direct = [], []
    if ready:
        stop = threading.Event()
        threading.Thread(target=poller, args=(stop,), daemon=True).start()
        t_gen = time.time()
        end = t_gen + A.minutes * 60
        summary["gen_start"] = el()
        with ThreadPoolExecutor(max_workers=A.conc) as pool:
            futs = [pool.submit(program_job, k, DATA[k], rep, end) for k in keys for rep in (0, 1)]
            futs += [pool.submit(direct_job, k, DATA[k], i, end) for k in keys for i in range(len(DATA[k]["solution"]))]
            for f in as_completed(futs):
                r = f.result()
                (prog if r["kind"] == "program" else direct).append(r)
        stop.set()
        summary["gen_seconds"] = round(time.time() - t_gen)
        rows = [json.loads(x) for x in (A.out / f"{A.tag}-metrics.jsonl").read_text().splitlines() if x.strip()]
        if len(rows) >= 2:
            g = "vllm:generation_tokens_total"
            summary["tok_per_s"] = round((rows[-1].get(g, 0) - rows[0].get(g, 0)) / max(1, rows[-1]["t"] - rows[0]["t"]), 1)
            summary["running_mean"] = round(sum(r.get("vllm:num_requests_running", 0) for r in rows) / len(rows), 1)
        summary.update(summarize(prog, direct))
    log(json.dumps({k: summary.get(k) for k in ("tok_per_s", "program", "direct")}))
    (A.out / f"{A.tag}-summary.json").write_text(json.dumps(summary, indent=1, default=str))
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
