import os
import sys
import time
import json
import argparse
import traceback
import torch
import torch.multiprocessing as mp


def estimated_work(task):
    """NVARC+-inspired rough cost proxy (train token burden + test decode burden)."""
    def ntok(grid):
        return len(grid) * (len(grid[0]) + 1)
    train_tokens = sum(ntok(p["input"]) + ntok(p["output"]) for p in task["train"])
    ratios = sorted(ntok(p["output"]) / max(1, ntok(p["input"])) for p in task["train"])
    ratio = ratios[len(ratios)//2]
    test_tokens = sum(ntok(t["input"]) * (1 + ratio) for t in task["test"])
    return train_tokens * 16 + test_tokens * 8 * len(task["test"])


def local_worker(rank, queue, end_time, marker_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.set_default_device("cpu")
    # The distinct marker directory is cleared before EACH new pass.
    if rank > 0:
        prev = os.path.join(marker_dir, f"worker{rank-1}")
        waited = 0
        while not os.path.exists(prev) and waited < 900:
            time.sleep(5)
            waited += 5
    try:
        from arc_solver import worker
        with open(os.path.join(marker_dir, f"worker{rank}"), "w") as f:
            f.write("Ok")
        print(f"[Rank {rank}] start!")
        worker(rank, queue, end_time)
        print(f"[Rank {rank}] done!")
    except Exception as e:
        print(f"[Rank {rank}] CRASH {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--keys-file", type=str, default="")
    parser.add_argument("--order", choices=["cheap", "sorted", "file"], default="cheap")
    args = parser.parse_args()
    rerun_mode = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    root = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
    if not os.path.isdir(root):
        root = "/kaggle/input/arc-prize-2026-arc-agi-2"
    test_path = os.path.join(root, "arc-agi_test_challenges.json" if rerun_mode else "arc-agi_evaluation_challenges.json")
    with open(test_path) as f:
        data = json.load(f)

    if args.keys_file:
        with open(args.keys_file) as f:
            keys = [k for k in json.load(f) if k in data]
    elif rerun_mode:
        keys = sorted(data)
    else:
        debug_keys = os.getenv("ARC_DEBUG_KEYS", "0934a4d8,36a08778,981571dc,aa4ec2a5").split(",")
        keys = [k for k in debug_keys if k in data]

    if args.order == "cheap":
        keys = sorted(keys, key=lambda k: (estimated_work(data[k]), k))
    elif args.order == "sorted":
        keys = sorted(keys)

    n_gpu = min(4, torch.cuda.device_count())
    if n_gpu == 0:
        raise RuntimeError("Four L4 GPUs required for Boyko TTT; no CUDA device found")
    marker_dir = os.path.join("/kaggle/working", "v16_markers", os.environ.get("ARC_RUN_TAG", "A"))
    os.makedirs(marker_dir, exist_ok=True)
    for name in os.listdir(marker_dir):
        try:
            os.remove(os.path.join(marker_dir, name))
        except OSError:
            pass
    print(f"[starter] tasks={len(keys)} | GPUs={n_gpu} | order={args.order} | "
          f"remaining_min={(args.end_time-time.time())/60:.1f} | output={os.getenv('ARC_OUT_DIR')}",flush=True)
    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    for _ in range(n_gpu):
        queue.put(None)
    mp.spawn(local_worker, args=(queue, args.end_time, marker_dir), nprocs=n_gpu)
