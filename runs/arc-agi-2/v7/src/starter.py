import argparse
import importlib
import json
import multiprocessing as mp
import os
import time

import arc_v6_config as cfg

MARKER_DIR = os.environ.get("ARC_V6_MARKER_DIR", "/kaggle")
SOLVER = os.environ.get("ARC_V6_SOLVER_MODULE", "arc_solver")  # test seam; production = arc_solver
MARKER_WAIT_SECONDS = 900
MARKER_POLL_SECONDS = float(os.environ.get("ARC_V6_MARKER_POLL", "5"))
KILL_GRACE_SECONDS = float(os.environ.get("ARC_V6_KILL_GRACE", "120"))


def marker(rank):
    return os.path.join(MARKER_DIR, f"worker{rank}")


def local_worker(rank, queue, end_time):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    if SOLVER == "arc_solver":
        import torch
        torch.set_default_device("cpu")

    # Fix Unsloth patching issue (author): import ranks one at a time, but never wait forever.
    if rank > 0:
        waited = 0
        while not os.path.exists(marker(rank - 1)) and waited < MARKER_WAIT_SECONDS:
            time.sleep(MARKER_POLL_SECONDS)
            waited += MARKER_POLL_SECONDS

    worker = importlib.import_module(SOLVER).worker

    with open(marker(rank), "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!", flush=True)

    worker(rank, queue, end_time)

    print(f"[Rank {rank}] done!", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    with open(cfg.work_challenges_path(), "r") as f:
        data = json.load(f)
    keys = cfg.select_keys(data)
    cfg.log("queue", rerun=cfg.is_rerun(), tasks=len(keys), challenges=len(data),
            seconds_left=round(args.end_time - time.time()))

    # v7: fixed hash seed so hash(bk)-seeded augmentation scoring is reproducible across runs.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    for key in keys:
        queue.put(key)
    for _ in range(cfg.NUM_WORKERS):
        queue.put(None)

    # Independent processes: one crashed GPU worker must not kill the other three
    # (torch.multiprocessing.spawn terminates all ranks when one fails).
    procs = [ctx.Process(target=local_worker, args=(rank, queue, args.end_time)) for rank in range(cfg.NUM_WORKERS)]
    for p in procs:
        p.start()
    while any(p.is_alive() for p in procs) and time.time() < args.end_time + KILL_GRACE_SECONDS:
        time.sleep(2)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(15)
        if p.is_alive():
            p.kill()
            p.join(5)
    cfg.log("workers_exited", exitcodes=[p.exitcode for p in procs])
    if any(p.is_alive() for p in procs):
        # An unkillable child would block multiprocessing's atexit join forever.
        cfg.log("abandoning_unkillable_workers")
        os._exit(0)
