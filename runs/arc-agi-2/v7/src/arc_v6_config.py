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
