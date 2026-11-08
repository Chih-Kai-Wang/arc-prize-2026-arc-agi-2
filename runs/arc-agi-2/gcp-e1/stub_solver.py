# CPU stand-in for arc_solver.worker (driver_e1 --solver_module stub_solver): same queue protocol, writes one
# decoded file per task atomically, raises on keys listed in STUB_FAIL, and simulates a preemption after STUB_STOP_AFTER tasks.
import bz2, os, pickle, sys, time
import arc_v6_config


def worker(rank, queue, end_time):
    out = arc_v6_config.outputs_dir()
    os.makedirs(out, exist_ok=True)
    fail = set(filter(None, os.environ.get("STUB_FAIL", "").split(",")))
    stop_after = int(os.environ.get("STUB_STOP_AFTER", "0"))
    n = 0
    while not queue.empty():
        key = queue.get()
        if key is None:
            break
        if stop_after and n == stop_after:
            # leave a partial file for the in-flight task, then die like a preempted VM
            open(os.path.join(out, f"{key}_0.partial"), "w").close()
            os._exit(9)
        try:
            if key in fail:
                raise RuntimeError("stub failure")
            tmp = os.path.join(out, f"._tmp_{key}_0.t0")
            with bz2.BZ2File(tmp, "w") as f:
                pickle.dump([{"solution": [[0]], "score": 1.0}], f)
            os.replace(tmp, os.path.join(out, f"{key}_0.t0"))
            print(f"[Rank {rank}] finished {key} in 0.1s", flush=True)
        except Exception as exc:
            print(f"[Rank {rank}] task {key} failed: {exc}", flush=True)
            arc_v6_config.mark_failed(key)
        n += 1
