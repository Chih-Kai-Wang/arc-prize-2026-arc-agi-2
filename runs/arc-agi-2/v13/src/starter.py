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


def marker(rank, label="A"):
    return os.path.join(MARKER_DIR, f"worker{rank}" + ("" if label == "A" else "." + label))


def local_worker(rank, queue, end_time, label="A", env=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ.update(env or {})  # v8: pass B seeds and output store, set before the solver is imported

    if SOLVER == "arc_solver":
        import torch
        torch.set_default_device("cpu")

    # Fix Unsloth patching issue (author): import ranks one at a time, but never wait forever.
    if rank > 0:
        waited = 0
        while not os.path.exists(marker(rank - 1, label)) and waited < MARKER_WAIT_SECONDS:
            time.sleep(MARKER_POLL_SECONDS)
            waited += MARKER_POLL_SECONDS

    worker = importlib.import_module(SOLVER).worker

    with open(marker(rank, label), "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!" + ("" if label == "A" else f" pass {label}"), flush=True)

    worker(rank, queue, end_time)

    print(f"[Rank {rank}] done!", flush=True)


def run_pass(ctx, manager, keys, end_time, label="A", env=None):
    """One pass: NUM_WORKERS independent GPU processes over a shared queue. True when every worker is gone."""
    queue = manager.Queue()
    for key in keys:
        queue.put(key)
    for _ in range(cfg.NUM_WORKERS):
        queue.put(None)

    # Independent processes: one crashed GPU worker must not kill the other three
    # (torch.multiprocessing.spawn terminates all ranks when one fails).
    procs = [ctx.Process(target=local_worker, args=(rank, queue, end_time, label, env)) for rank in range(cfg.NUM_WORKERS)]
    for p in procs:
        p.start()
    while any(p.is_alive() for p in procs) and time.time() < end_time + KILL_GRACE_SECONDS:
        time.sleep(2)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(15)
        if p.is_alive():
            p.kill()
            p.join(5)
    cfg.log("workers_exited", exitcodes=[p.exitcode for p in procs], pass_label=label)
    return not any(p.is_alive() for p in procs)


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

    clean = run_pass(ctx, manager, keys, args.end_time)

    # v8: pass B only in wall time pass A did not need; a saturated run behaves exactly like v7.
    end_b = args.end_time - cfg.PASS_B_DEADLINE_MARGIN
    seconds_left = end_b - time.time()
    if clean and seconds_left >= cfg.PASS_B_MIN_SECONDS:
        keys_b, plan = cfg.pass_b_plan(keys, cfg.outputs_dir(), data, seconds_left)  # v9: unsure tasks, near-full caps
        env_b = dict(cfg.PASS_B_ENV, ARC_V6_OUTPUTS=cfg.outputs_dir_b())
        missing = len(set(keys) - cfg.covered_tasks(cfg.outputs_dir()))
        cfg.log("pass_b_start", tasks=len(keys_b), missing_after_a=missing,
                failed_after_a=len(cfg.failed_tasks(cfg.outputs_dir())), seconds_left=round(end_b - time.time()), **plan)
        clean = run_pass(ctx, manager, keys_b, end_b, label="B", env=env_b)
    else:
        cfg.log("pass_b_skipped", clean=clean, seconds_left=round(seconds_left))
    if not clean:
        # An unkillable child would block multiprocessing's atexit join forever.
        cfg.log("abandoning_unkillable_workers")
        os._exit(0)
