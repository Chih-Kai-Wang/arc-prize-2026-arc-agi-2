"""Program-synthesis channel helpers (CSR-arc-agi-2-001 #1/#3, HANDOFF 10-13): prompt for a `transform(grid)` program,
extract it, run it in a resource-limited subprocess on the train inputs and the test inputs, and build one round of
execution feedback. Pure stdlib; the sandbox is a separate `python -I` process per program."""
import json
import re
import subprocess
import sys
import textwrap

import arc_llm as L

RUNNER = r'''
import json, resource, sys
for lim, val in ((resource.RLIMIT_AS, 2 << 30), (resource.RLIMIT_CPU, 20)):
    try:
        resource.setrlimit(lim, (val, val))
    except (ValueError, OSError):  # macOS refuses RLIMIT_AS; Linux (Colab/Kaggle) applies it
        pass
src, inputs = json.loads(sys.stdin.read())
ns = {}
out = []
try:
    exec(compile(src, "<model>", "exec"), ns)
    fn = ns["transform"]
except Exception as e:
    print(json.dumps({"compile_error": repr(e)[:400]})); sys.exit(0)
for g in inputs:
    try:
        r = fn([list(row) for row in g])
        if hasattr(r, "tolist"):
            r = r.tolist()
        r = [[int(v) for v in row] for row in r]
        out.append({"ok": r})
    except Exception as e:
        out.append({"err": repr(e)[:300]})
print(json.dumps({"results": out}))
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
    """Returns (results, error): results is a list of {"ok": grid} / {"err": msg} per input grid."""
    try:
        p = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=json.dumps([code, grids]), text=True,
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return None, ("no output: " + (p.stderr or "")[-300:])
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
