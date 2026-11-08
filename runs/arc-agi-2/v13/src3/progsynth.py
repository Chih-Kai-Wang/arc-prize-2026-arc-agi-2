"""Program-synthesis channel helpers (CSR-arc-agi-2-001 #1/#3, HANDOFF 10-13; v13: sandbox hardened per API review 20261013T150520Z): prompt for a `transform(grid)` program,
extract it, run it in a resource-limited subprocess on the train inputs and the test inputs, and build one round of
execution feedback. Pure stdlib; the sandbox is a separate `python -I` process per program."""
import json
import re
import subprocess
import sys
import textwrap

import arc_llm as L

RUNNER = r'''
import json, os, resource, sys
for lim, val in ((resource.RLIMIT_AS, 2 << 30), (resource.RLIMIT_CPU, 20), (resource.RLIMIT_FSIZE, 10 << 20)):
    try:
        resource.setrlimit(lim, (val, val))
    except (ValueError, OSError):  # macOS refuses RLIMIT_AS; Linux (Colab/Kaggle) applies it
        pass
real = os.fdopen(os.dup(1), "w")  # the result channel; the model's own prints go to /dev/null (no parent OOM)
null = open(os.devnull, "w")
os.dup2(null.fileno(), 1)
os.dup2(null.fileno(), 2)
sys.stdout = sys.stderr = null
src, inputs = json.loads(sys.stdin.read())
ns = {}
out = []
try:
    exec(compile(src, "<model>", "exec"), ns)
    fn = ns["transform"]
except BaseException as e:
    real.write(json.dumps({"compile_error": repr(e)[:400]})); real.flush(); os._exit(0)
for g in inputs:
    try:
        r = fn([list(row) for row in g])
        if hasattr(r, "tolist"):
            r = r.tolist()
        r = [[int(v) for v in row] for row in r]
        out.append({"ok": r})
    except BaseException as e:
        out.append({"err": repr(e)[:300]})
real.write(json.dumps({"results": out})); real.flush()
os._exit(0)
'''


def build_prompt(task):
    parts = ["You are solving an ARC-AGI puzzle by writing a program. Each example maps an input grid to an output grid "
             "with the same hidden rule. Grids are lists of rows; each cell is a colour 0-9.", ""]
    for i, p in enumerate(task["train"], 1):
        parts += [f"Example {i} input:", L.grid_text(p["input"]), f"Example {i} output:", L.grid_text(p["output"]), ""]
    for i, p in enumerate(task["test"], 1):
        parts += [f"Test input {i} (output unknown):", L.grid_text(p["input"]), ""]
    parts += ["Write a Python function `transform(grid)` that takes one input grid (list of lists of int) and returns "
              "the output grid (list of lists of int). It must reproduce every example output exactly and generalise "
              "to the test inputs. You may use the standard library and numpy. Give the complete final code in one "
              "```python code block at the end of your answer."]
    return "\n".join(parts)


_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    blocks = [b for b in _CODE.findall(text or "") if "def transform" in b]
    return blocks[-1] if blocks else None


def run_program(code, grids, timeout=30):
    """Returns (results, error): results is a list of {"ok": grid} / {"err": msg} per input grid. The child runs in its
    own session and an empty temp cwd, single-threaded BLAS, and is killed with its whole process group on timeout."""
    import os
    import shutil
    import signal
    import tempfile
    cwd = tempfile.mkdtemp(prefix="v13prog-")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = None
    try:
        # -s, not -I: keep the notebook's PYTHONPATH (Kaggle's numpy may live there); the cwd is an empty temp dir, so
        # nothing can be planted next to the runner
        p = subprocess.Popen([sys.executable, "-s", "-c", RUNNER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, cwd=cwd, env=env, start_new_session=True)
        stdout, _ = p.communicate(json.dumps([code, grids]), timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
        p.communicate()
        return None, "timeout"
    finally:
        if p is not None and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(cwd, ignore_errors=True)
    try:
        out = json.loads(stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return None, "no output (killed or crashed)"
    if "compile_error" in out:
        return None, out["compile_error"]
    return out["results"], None


def check(code, task):
    """Run on train + test inputs of the (augmented) task. verified = every train output reproduced exactly and every
    test output is a valid grid."""
    grids = [p["input"] for p in task["train"]] + [p["input"] for p in task["test"]]
    res, err = run_program(code, grids)
    n = len(task["train"])
    if res is None:
        return {"verified": False, "error": err, "train_pass": 0, "train_n": n, "tests": None}
    passes = [r.get("ok") == p["output"] for r, p in zip(res[:n], task["train"])]
    tests = [r.get("ok") if L.valid_grid(r.get("ok")) else None for r in res[n:]]
    return {"verified": all(passes) and all(t is not None for t in tests), "error": None, "train_pass": sum(passes),
            "train_n": n, "tests": tests, "train_results": res[:n]}


def feedback(task, chk):
    """One round of execution feedback for a failed program."""
    if chk.get("error"):
        return textwrap.shorten(f"Your code failed to run: {chk['error']}", 600) + \
            "\nFix it. Give the complete corrected code in one ```python block at the end."
    lines = [f"Your program reproduces {chk['train_pass']} of {chk['train_n']} example outputs. Mismatches:"]
    for i, (r, p) in enumerate(zip(chk["train_results"], task["train"]), 1):
        if r.get("ok") == p["output"]:
            continue
        if "err" in r:
            lines.append(f"Example {i}: error {r['err'][:200]}")
            continue
        got = r.get("ok")
        if not L.valid_grid(got) or len(got) * len(got[0]) > 400 or len(p["output"]) * len(p["output"][0]) > 400:
            shape = f"{len(got)}x{len(got[0]) if got and isinstance(got[0], list) else '?'}" if isinstance(got, list) else "?"
            lines.append(f"Example {i}: expected shape {len(p['output'])}x{len(p['output'][0])}, got {shape}")
        else:
            lines.append(f"Example {i}: expected\n{L.grid_text(p['output'])}\ngot\n{L.grid_text(got)}")
    lines.append("Find the rule that fits every example, then give the complete corrected code in one ```python block "
                 "at the end.")
    return "\n".join(lines)
