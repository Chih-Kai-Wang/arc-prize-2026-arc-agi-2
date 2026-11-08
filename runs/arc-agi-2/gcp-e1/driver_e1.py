# E1 one-GPU replay (artifacts/verification/arc-agi-2/gcp-1xl4-spec-v1.md): one (src, pass, seed set) run over keys30
# with a fixed per-task cap and per-task done markers, so a later VM resumes where a preempted one stopped.
import argparse, json, os, queue, sys, time
p = argparse.ArgumentParser()
p.add_argument("--src", required=True)                       # runs/arc-agi-2/v8/src (arm F) or v11/src (arm H)
p.add_argument("--pass_label", choices=["A", "B"], default="A")
p.add_argument("--seed_offset", type=int, default=0)        # s1 0, s2 1000, s3 2000 (prereg)
p.add_argument("--cap", type=float, required=True)          # F 1200, H 800
p.add_argument("--keys", required=True)
p.add_argument("--run_dir", required=True)                  # <run_dir>/store, <run_dir>/done/<key>
p.add_argument("--hours", type=float, default=23.0)
p.add_argument("--solver_module", default="arc_solver")     # test seam: CPU stub
a = p.parse_args()
sys.path.insert(0, a.src)
import arc_v6_config as cfg

A_SEEDS = {"ARC_V8_TRAIN_AUG_SEED": 1, "ARC_V8_EVAL_AUG_SEED": 2, "ARC_V8_TRAINER_SEED": 42,
           "ARC_V8_LORA_SEED": 42, "ARC_V8_SCORE_SEED_OFFSET": 0}  # code defaults of pass A (arc_solver.py)
base = A_SEEDS if a.pass_label == "A" else {k: int(v) for k, v in cfg.PASS_B_ENV.items()}
seeds = {k: str(v + a.seed_offset) for k, v in base.items()}
os.environ.update(seeds)
store, done_dir = os.path.join(a.run_dir, "store"), os.path.join(a.run_dir, "done")
os.makedirs(store, exist_ok=True)
os.makedirs(done_dir, exist_ok=True)
os.environ["ARC_V6_OUTPUTS"] = store
cfg.adaptive_cap = lambda remaining, pending: a.cap  # fixed cap: no global deadline, so per-task resume is valid

keys = json.load(open(a.keys))
done = set(os.listdir(done_dir))
todo = [k for k in keys if k not in done]
# A task cut off mid-decode left some of its files behind; drop them so the rerun's store matches a clean run.
stale = [f for f in os.listdir(store) if cfg.output_task(f.replace("._tmp_", "")) in todo]
for f in stale:
    os.remove(os.path.join(store, f))


class MarkingQueue(queue.Queue):
    """The worker asks for the next key only after the previous task returned (solved or recorded as failed)."""
    prev = None

    def get(self, *args, **kwargs):
        if self.prev is not None:
            open(os.path.join(done_dir, self.prev), "w").close()
        self.prev = super().get(*args, **kwargs)
        return self.prev


q = MarkingQueue()
for k in todo:
    q.put(k)
q.put(None)
try:
    import torch
    torch.set_default_device("cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
except ImportError:  # only the CPU stub test runs without torch; arc_solver itself imports it
    gpu = "none"
solver = __import__(a.solver_module)
t0 = time.time()
cfg.log("driver_start", src=a.src, pass_label=a.pass_label, seed_offset=a.seed_offset, seeds=seeds, cap=a.cap,
        tasks=len(keys), resumed_done=len(done), todo=len(todo), stale_files_removed=len(stale), gpu=gpu)
solver.worker(0, q, t0 + a.hours * 3600)
finished = len(os.listdir(done_dir))
cfg.log("driver_done", seconds=round(time.time() - t0), done=finished, of=len(keys))
sys.exit(0 if finished == len(keys) else 3)
