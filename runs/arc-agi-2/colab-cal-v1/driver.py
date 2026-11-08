# One-GPU replay of the v8 solver worker on a sample of public-evaluation tasks (rerun-mode caps).
import argparse, json, os, queue, sys, time
p = argparse.ArgumentParser(); p.add_argument("--pass_label", default="A"); p.add_argument("--hours", type=float, default=3.0)
a = p.parse_args()
sys.path.insert(0, "/content/src")
import arc_v6_config as cfg
if a.pass_label == "B":
    os.environ.update(cfg.PASS_B_ENV)
os.environ["ARC_V6_OUTPUTS"] = "/content/out_" + a.pass_label.lower()
cfg.adaptive_cap = lambda remaining, pending: cfg.TASK_SECONDS   # unsaturated-rerun regime: author cap 1200 s
import torch
torch.set_default_device("cpu")
import arc_solver
keys = json.load(open("/content/sample_keys.json"))
q = queue.Queue()
for k in keys:
    q.put(k)
q.put(None)
t0 = time.time()
cfg.log("driver_start", pass_label=a.pass_label, tasks=len(keys), hours=a.hours, gpu=torch.cuda.get_device_name(0),
        seeds={k: os.environ.get(k) for k in cfg.PASS_B_ENV})
arc_solver.worker(0, q, t0 + a.hours * 3600)
cfg.log("driver_done", pass_label=a.pass_label, seconds=round(time.time() - t0), left_in_queue=q.qsize())
