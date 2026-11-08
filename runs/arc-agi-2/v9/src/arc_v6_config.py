"""v6 runtime configuration: path discovery, rerun gating, task selection, deadlines.

No task IDs, hashes or counts of the public test file are pinned. Rerun mode takes every
task in the mounted test file; commit mode smoke-tests a few evaluation tasks chosen by
observed size only.
"""
import glob
import json
import os
import time

COMPETITION_SLUG = "arc-prize-2026-arc-agi-2"
MODEL_KEY = "qwen3_4b_grids15_sft139"
INPUT_ROOT = os.environ.get("ARC_V6_INPUT_ROOT", "/kaggle/input")
WORK_DIR = os.environ.get("ARC_V6_WORK_DIR", "/kaggle/working")
RERUN_WALL_SECONDS = 12 * 3600 - 600   # author global_end_time margin
COMMIT_WALL_SECONDS = 3600             # 4-task smoke; bounds the charged commit run
TASK_SECONDS = 1200                    # author per-task decode cap
TASK_FLOOR_SECONDS = 360               # v7: adaptive cap never below this (TTT + one decode batch)
COMMIT_TASK_SECONDS = 900              # v7: lower commit-smoke cap so the <TASK_SECONDS branch runs on hardware
SMOKE_TASKS = 4
NUM_WORKERS = 4
# v8: pass B re-runs tasks with independent seeds in the wall time pass A leaves unused; its candidates are
# pooled at assembly with pass A's top-1 always kept. Pass A itself is unchanged from v7.
PASS_B_MIN_SECONDS = float(os.environ.get("ARC_V8_PASS_B_MIN_SECONDS", "1200"))  # reload + floor + a task
# Pass B ends this much before end_time, so its kill + the two-store assembly stay well inside the 12 h limit.
PASS_B_DEADLINE_MARGIN = float(os.environ.get("ARC_V8_PASS_B_DEADLINE_MARGIN", "300"))
PASS_B_ENV = {"ARC_V8_TRAIN_AUG_SEED": "17", "ARC_V8_EVAL_AUG_SEED": "29", "ARC_V8_TRAINER_SEED": "137",
              "ARC_V8_LORA_SEED": "137", "ARC_V8_SCORE_SEED_OFFSET": "7"}
# v9: pass B first redraws the tasks pass A is unsure of that a floor cap would have cut, and only as many as
# their predicted full time fits (Colab pairing 2026-10-06: a second seed at cap 1200 solved a mid-size task pass A
# missed; at the 360 s floor it never did; tasks under ~360 s already got full redraws in v8).
PASS_B_CONFIDENT = float(os.environ.get("ARC_V9_PASS_B_CONFIDENT", "5"))  # kgmon top-1 score counted as sure
PASS_B_PLAN_SECONDS = float(os.environ.get("ARC_V9_PASS_B_PLAN_SECONDS", "120"))  # bound on reading store A
PASS_B_PLAN_FACTOR = 1.0    # v6-v8 Kaggle 4xL4 smoke task times were 0.5-1.0x this one-L4 fit (no contention slowdown)
PASS_B_START_STAGGER = 180  # pass-B ranks import one at a time (~44 s each) before taking tasks
PASS_B_PLAN_SCALE = float(os.environ.get("ARC_V9_PASS_B_PLAN_SCALE", "1"))  # test seam only; production = 1
SECONDS_FIT = (-223.08392554483123, 0.3000408772151657, -1.550904629974504e-05)  # colab-cal-v1 one-L4, cap 1200


def is_rerun():
    # Same truthiness as the author notebook.
    return bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))


def _single(paths, prefer, what):
    paths = sorted(set(paths))
    preferred = [p for p in paths if prefer in p.lower()]
    paths = preferred or paths
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one {what}, found {len(paths)}: {paths[:5]}")
    return paths[0]


def competition_dir(root=None):
    root = root or INPUT_ROOT
    hits = glob.glob(os.path.join(root, "**", "arc-agi_test_challenges.json"), recursive=True)
    return os.path.dirname(_single(hits, COMPETITION_SLUG, "competition test file"))


def model_dir(root=None):
    root = root or INPUT_ROOT
    hits = [os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "config.json"), recursive=True)
            if MODEL_KEY in p.lower()]
    return _single(hits, MODEL_KEY, "model directory")


def test_path(root=None):
    return os.path.join(competition_dir(root), "arc-agi_test_challenges.json")


def work_challenges_path(root=None):
    """Challenge file the model workers process in this mode."""
    name = "arc-agi_test_challenges.json" if is_rerun() else "arc-agi_evaluation_challenges.json"
    return os.path.join(competition_dir(root), name)


def _cells(task):
    return sum(len(g) * len(g[0]) for pairs in task.values() for pair in pairs for g in pair.values() if g)


def select_keys(challenges, rerun=None):
    """Rerun: every task, cheapest estimated work first (v7). Commit: size-chosen smoke set."""
    rerun = is_rerun() if rerun is None else rerun
    keys = sorted(challenges)
    if rerun:
        return sorted(keys, key=lambda k: (estimated_work(challenges[k]), k))
    by_size = sorted(keys, key=lambda k: (_cells(challenges[k]), k))
    chosen = [by_size[-1]]                                   # largest context: VRAM/length check
    multi = [k for k in by_size if len(challenges[k]["test"]) > 1 and k not in chosen]
    chosen += multi[:1]                                      # multiple test inputs
    chosen += [k for k in by_size if k not in chosen][:SMOKE_TASKS - len(chosen)]
    return sorted(chosen)


def estimated_work(task):
    """Size proxy: TTT cost grows with all grid cells, decoding with the number of test inputs."""
    try:
        return _cells(task) * (1 + 0.5 * (len(task["test"]) - 1))
    except Exception:  # an unexpected task shape must never stop the queue from being built
        return 0


def adaptive_cap(remaining_seconds, pending_tasks):
    """Fair share of the remaining wall over pending + in-flight tasks, within [floor, author cap]."""
    share = remaining_seconds * NUM_WORKERS / (max(0, pending_tasks) + NUM_WORKERS)
    upper = TASK_SECONDS if is_rerun() else COMMIT_TASK_SECONDS
    return max(TASK_FLOOR_SECONDS, min(upper, share))


def seed(name, default):
    """v8: per-pass seed; unset (pass A) keeps the v7 value."""
    return int(os.environ.get(name, default))


def outputs_dir():
    return os.environ.get("ARC_V6_OUTPUTS", "/kaggle/inference_outputs")


def outputs_dir_b():
    return os.environ.get("ARC_V8_OUTPUTS_B", outputs_dir().rstrip("/") + "_b")


def output_task(name):
    """Task key of a decoded-result file name ('<task>_<i>.<augmentation>')."""
    return name.split(".")[0].rsplit("_", 1)[0]


def covered_tasks(outputs):
    """Tasks with at least one finished decoded-result file in a store."""
    try:
        names = os.listdir(outputs)
    except OSError:
        names = []
    return {output_task(n) for n in names if not n.startswith("._tmp_")}


def failed_path(outputs):
    return outputs.rstrip("/") + "_failed.txt"


def mark_failed(key):
    """v8: record a task whose solve raised, so pass B tries it last; never raises."""
    try:
        with open(failed_path(outputs_dir()), "a") as f:
            f.write(json.dumps(key) + "\n")
    except Exception:
        pass


def failed_tasks(outputs):
    try:
        with open(failed_path(outputs)) as f:
            return {json.loads(line) for line in f if line.strip()}
    except Exception:
        return set()


def pass_b_keys(keys, outputs_a):
    """Pass B order (pass-A order kept within groups): tasks pass A left without any decoded result and that
    did not raise, then tasks with results, then tasks that raised without results (likely to fail again)."""
    covered, failed = covered_tasks(outputs_a), failed_tasks(outputs_a)
    missing = [k for k in keys if k not in covered and k not in failed]
    return missing + [k for k in keys if k in covered] + [k for k in keys if k not in covered and k in failed]


def task_confidence(keys, outputs_a, challenges, deadline):
    """v9: per task, the minimum over its test outputs of pass A's kgmon top-1 score; an output without any
    readable candidate counts as -inf (unsure). Raises TimeoutError past the deadline."""
    import bz2
    import pickle
    from arc_decoder import getter_kgmon, hashable
    groups = {}
    for name in sorted(os.listdir(outputs_a)):
        if time.time() > deadline:
            raise TimeoutError("reading pass-A store took too long")
        if name.startswith("._tmp_"):
            continue
        try:
            with bz2.BZ2File(os.path.join(outputs_a, name)) as f:
                samples = pickle.load(f)
        except Exception:
            continue  # same tolerance as the assembly loader
        by_grid = groups.setdefault(name.split(".")[0], {})
        for sample in samples:
            try:
                by_grid.setdefault(hashable(sample["solution"]), []).append(sample)
            except Exception:
                pass
    conf = {}
    for k in keys:
        scores = []
        for i in range(len(challenges[k]["test"])):
            best = float("-inf")
            for guesses in groups.get(f"{k}_{i}", {}).values():
                try:
                    best = max(best, float(getter_kgmon(guesses)))
                except Exception:
                    pass
            scores.append(best)
        conf[k] = min(scores) if scores else float("-inf")
    return conf


def task_tokens(task):
    """Prompt length the model sees (colab-cal-v1 analyze.tokens): train pairs + the largest test input + 900."""
    gl = lambda g: len(g) * (len(g[0]) + 1)
    return min(8192, sum(gl(p["input"]) + gl(p["output"]) + 10 for p in task["train"])
               + max(gl(q["input"]) for q in task["test"]) + 900)


def full_seconds(task):
    """v9: predicted Kaggle seconds for one uncut solve (TTT + decode); an odd shape plans the author cap."""
    try:
        L = task_tokens(task)
        return (SECONDS_FIT[0] + SECONDS_FIT[1] * L + SECONDS_FIT[2] * L * L) * PASS_B_PLAN_FACTOR
    except Exception:
        return float(TASK_SECONDS)


def pass_b_plan(keys, outputs_a, challenges, seconds_left):
    """v9 pass-B queue, pass-A cheap-first order kept within every group: tasks pass A left without results; tasks
    pass A is unsure of whose full solve a floor cap would cut; the other unsure tasks (v8 already redrew those
    fully); sure tasks; tasks that raised. Truncated where the planned seconds (full solve, clipped to the
    [floor, author] caps) exceed the pass-B worker budget. Any error falls back to v8's full queue (pass_b_keys).
    Returns (keys, info)."""
    start = time.time()
    try:
        conf = task_confidence(keys, outputs_a, challenges, start + PASS_B_PLAN_SECONDS)
        covered, failed = covered_tasks(outputs_a), failed_tasks(outputs_a)
        full = {k: full_seconds(challenges[k]) for k in keys}
        missing = [k for k in keys if k not in covered and k not in failed]
        unsure = [k for k in keys if k in covered and not conf[k] >= PASS_B_CONFIDENT]
        long_ = [k for k in unsure if full[k] > TASK_FLOOR_SECONDS]
        short = [k for k in unsure if not full[k] > TASK_FLOOR_SECONDS]
        sure = [k for k in keys if k in covered and conf[k] >= PASS_B_CONFIDENT]
        raised = [k for k in keys if k not in covered and k in failed]
        budget = (seconds_left - (time.time() - start) - PASS_B_START_STAGGER * PASS_B_PLAN_SCALE) * NUM_WORKERS
        ordered, planned, total = missing + long_ + short + sure + raised, [], 0.0
        for k in ordered:
            need = min(TASK_SECONDS, max(TASK_FLOOR_SECONDS, full[k])) * PASS_B_PLAN_SCALE
            if len(planned) >= NUM_WORKERS and total + need > budget:
                break
            planned.append(k)
            total += need
        info = {"pass_b_order": "confidence", "missing": len(missing), "unsure_long": len(long_),
                "unsure_short": len(short), "sure": len(sure), "raised": len(raised), "limit": len(planned),
                "planned_sum": round(total), "budget": round(budget), "threshold": PASS_B_CONFIDENT,
                "plan_seconds": round(time.time() - start, 1)}
        return planned, info
    except Exception as exc:
        return pass_b_keys(keys, outputs_a), {"pass_b_order": "fallback", "error": f"{type(exc).__name__}: {exc}"[:300],
                                              "plan_seconds": round(time.time() - start, 1)}


def end_time(start):
    return start + (RERUN_WALL_SECONDS if is_rerun() else COMMIT_WALL_SECONDS)


def atomic_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f)
    os.replace(tmp, path)


def fallback_submission(test_challenges):
    return {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in task["test"]]
            for k, task in test_challenges.items()}


def validate_submission(test_challenges, submission):
    if set(submission) != set(test_challenges):
        raise ValueError("submission keys differ from test keys")
    for k, task in test_challenges.items():
        entries = submission[k]
        if not isinstance(entries, list) or len(entries) != len(task["test"]):
            raise ValueError(f"{k}: wrong number of test entries")
        for e in entries:
            if not isinstance(e, dict) or set(e) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"{k}: attempts missing")
            if not all(valid_grid(g) for g in e.values()):
                raise ValueError(f"{k}: invalid grid")
    return True


def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and isinstance(g[0], list) and 1 <= len(g[0]) <= 30
            and all(isinstance(r, list) and len(r) == len(g[0]) for r in g)
            and all(type(c) is int and 0 <= c <= 9 for r in g for c in r))


def log(message, **fields):
    print(json.dumps({"t": round(time.time(), 1), "msg": message, **fields}), flush=True)
