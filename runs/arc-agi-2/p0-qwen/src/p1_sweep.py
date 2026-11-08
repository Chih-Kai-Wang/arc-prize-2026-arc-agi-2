"""P1: serving-config throughput sweep for Qwen3.8-27B-FP8 on Kaggle 4xL4 (follow-up to P0, 2026-10-13).

P0 measured ~280 tok/s at TP=4, MTP=2, 16 streams, async scheduling OFF; ~133 ms per decode step, far above the
weight-bandwidth floor, so per-step overhead (scheduling, TP all-reduce over PCIe, kernels) dominates. This run measures
the same xhigh ARC workload under several configs, each for a fixed window after warm-up, and reports aggregate and
per-stream tok/s, spec-decode acceptance and KV use. Reuses the P0 helpers (install, server env, metrics, requests)."""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("P0_OUT", "/kaggle/working/p1")
os.environ.setdefault("P0_WALL_SECONDS", str(2 * 3600 + 5 * 60))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_probe as P  # noqa: E402

MEASURE = int(os.environ.get("P1_MEASURE_SECONDS", 720))
PORT = 8000
CONFIGS = [  # most informative first; later ones are dropped if time runs out
    {"name": "base_tp4_mtp2_c16_noasync", "tp": 4, "mtp": 2, "async": False, "conc": 16},
    {"name": "tp4_mtp2_c16_async", "tp": 4, "mtp": 2, "async": True, "conc": 16},
    {"name": "tp4_mtp2_c32_async", "tp": 4, "mtp": 2, "async": True, "conc": 32},
    {"name": "tp2_mtp2_c16_async_64k", "tp": 2, "gpus": "0,1", "mtp": 2, "async": True, "conc": 16,
     "max_len": 65536, "util": 0.95},
    {"name": "tp4_mtp3_c32_async", "tp": 4, "mtp": 3, "async": True, "conc": 32},
]
R = {"version": "p1-v1", "measure_seconds": MEASURE, "configs": {}}


def save():
    (P.OUT / "p1-summary.json").write_text(json.dumps(R, indent=1, default=str))


def start(model, cfg, eager=False):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
           "--served-model-name", "q38", "--host", "127.0.0.1", "--port", str(PORT),
           "--tensor-parallel-size", str(cfg["tp"]), "--gpu-memory-utilization", str(cfg.get("util", 0.93)),
           "--max-model-len", str(cfg.get("max_len", P.MAX_LEN)), "--max-num-seqs", str(max(48, cfg["conc"])),
           "--kv-cache-dtype", "fp8", "--attention-backend", "TRITON_ATTN", "--reasoning-parser", "qwen3",
           "--trust-remote-code", "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if not cfg["async"]:
        cmd += ["--no-async-scheduling"]
    if cfg["mtp"]:
        cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": cfg["mtp"]})]
    if eager:
        cmd += ["--enforce-eager"]
    path = P.OUT / f"server-{cfg['name']}{'-eager' if eager else ''}.log"
    proc = P.subprocess.Popen(cmd, stdout=path.open("w"), stderr=P.subprocess.STDOUT, stdin=P.subprocess.DEVNULL,
                              text=True, start_new_session=True, env=P.server_env(cfg.get("gpus")))
    return proc, path


def load(cfg, jobs, end):
    """Keep cfg['conc'] streams busy until `end`, cycling through the xhigh prompts."""
    nxt = iter(range(10 ** 9))
    lock = threading.Lock()

    def worker():
        while time.time() < end:
            with lock:
                j = dict(jobs[next(nxt) % len(jobs)], phase=cfg["name"])
            P.one_request(j, PORT, end, stop_at=end)

    with ThreadPoolExecutor(max_workers=cfg["conc"]) as pool:
        for _ in range(cfg["conc"]):
            pool.submit(worker)


def run_config(model, jobs, cfg):
    rec = R["configs"][cfg["name"]] = {"cfg": cfg}
    need = MEASURE + 20 * 60
    if P.T0 + P.WALL - time.time() < need:
        rec["skipped"] = "not enough time"
        save()
        return
    for eager in (False, True):
        if not cfg["async"] and eager:
            break
        proc, path = start(model, cfg, eager)
        t = time.time()
        ready = P.wait_ready(proc, PORT, min(t + 20 * 60, P.T0 + P.WALL - MEASURE - 5 * 60))
        rec.update(eager=eager, ready=ready, startup_seconds=round(time.time() - t), **P.kv_capacity(path))
        if ready and P.smoke(PORT)[0]:
            break
        rec["log_tail"] = path.read_text(errors="ignore")[-2500:] if path.exists() else ""
        P.kill(proc)
        proc = None
        if cfg["tp"] == 4:  # eager fallback only for the TP=2 memory case
            break
    if not proc:
        P.log(f"{cfg['name']}: server failed")
        save()
        return
    stop = threading.Event()
    th = threading.Thread(target=P.poller, args=(PORT, stop, cfg["name"]), daemon=True)
    t_start = P.el()
    th.start()
    load(cfg, jobs, time.time() + MEASURE)
    stop.set()
    th.join(timeout=30)
    rec["throughput"] = P.throughput(cfg["name"], t_start + 120 - 300, P.el(), cfg["conc"])  # loaded window from 2 min
    P.log(f"{cfg['name']}: {json.dumps(rec['throughput'])}")
    save()
    P.kill(proc)


def main():
    P.OUT.mkdir(parents=True, exist_ok=True)
    try:
        comp, model, wheels = P.phase_env()
        P.verify_wheels(wheels)
        P.install(wheels)
        _, jobs = P.build_jobs(comp)
        jobs = [j for j in jobs if j["effort"] == "xhigh"]
        for cfg in CONFIGS:
            run_config(model, jobs, cfg)
    except SystemExit as e:
        R["fatal"] = str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        R["fatal"] = repr(e)
        R["traceback"] = traceback.format_exc()[-3000:]
    finally:
        P.ABORT.set()
        R["phases"] = P.S["phases"]
        R["elapsed_seconds"] = P.el()
        save()
        P.log("p1 finished")


if __name__ == "__main__":
    main()
