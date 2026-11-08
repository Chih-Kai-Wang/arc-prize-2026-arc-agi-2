"""Pure helpers for the Qwen3.8 ARC probe (new-approach-proposal-v1 §6): prompt, answer parsing, dihedral + colour
augmentation and its inverse, and summary statistics. No I/O, no third-party imports, so it runs in the probe and in
local unit tests alike."""
import json
import math
import random
import re
import zlib

# --- grids -----------------------------------------------------------------------------------------------------------

def valid_grid(g):
    return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) for r in g)
            and 0 < len(g[0]) <= 30 and all(len(r) == len(g[0]) for r in g)
            and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))


def transpose(g):
    return [list(r) for r in zip(*g)]


def rot90(g):  # clockwise
    return [list(r) for r in zip(*g[::-1])]


def dihedral(g, k):
    """k in 0..7: rotate k%4 quarter turns clockwise, then transpose if k >= 4."""
    for _ in range(k % 4):
        g = rot90(g)
    return transpose(g) if k >= 4 else [list(r) for r in g]


def dihedral_inverse(g, k):
    if k >= 4:
        g = transpose(g)
    for _ in range((4 - k % 4) % 4):
        g = rot90(g)
    return [list(r) for r in g]


def recolor(g, perm):
    return [[perm[v] for v in r] for r in g]


def augmentation(task_id, salt="p0-v1"):
    """Deterministic non-identity transform per task: dihedral k in 1..7 and a permutation of colours 1..9 (0 fixed,
    ARC background)."""
    rng = random.Random(zlib.crc32(f"{salt}:{task_id}".encode()))
    k = rng.randrange(1, 8)
    rest = list(range(1, 10))
    rng.shuffle(rest)
    perm = [0] + rest
    return {"k": k, "perm": perm}


def apply_aug(grid, aug):
    return recolor(dihedral(grid, aug["k"]), aug["perm"])


def invert_aug(grid, aug):
    inv = [0] * 10
    for a, b in enumerate(aug["perm"]):
        inv[b] = a
    return dihedral_inverse(recolor(grid, inv), aug["k"])


def augment_task(task, aug):
    return {"train": [{"input": apply_aug(p["input"], aug), "output": apply_aug(p["output"], aug)}
                      for p in task["train"]],
            "test": [{"input": apply_aug(p["input"], aug)} for p in task["test"]]}


# --- prompt / parse --------------------------------------------------------------------------------------------------

def grid_text(g):
    return "[" + ",\n ".join(json.dumps(r, separators=(",", ":")) for r in g) + "]"


def build_prompt(task, test_index):
    parts = ["You are solving an ARC-AGI puzzle. Each example maps an input grid to an output grid with the same "
             "hidden rule. Grids are lists of rows; each cell is a colour 0-9.", ""]
    for i, p in enumerate(task["train"], 1):
        parts += [f"Example {i} input:", grid_text(p["input"]), f"Example {i} output:", grid_text(p["output"]), ""]
    parts += ["Test input:", grid_text(task["test"][test_index]["input"]), "",
              "Find the rule, apply it to the test input, and give the test output grid. End your answer with the "
              "output grid as a JSON list of lists inside a ```json code block, and nothing after it."]
    return "\n".join(parts)


_FENCE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.S)


def _candidates(text):
    for m in _FENCE.finditer(text):
        yield m.group(1)
    # bare outermost [[...]] blocks, last ones last
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]


def parse_answer(text):
    """Last valid grid found, scanning fenced blocks first and then every bare outermost list, so a valid bare list after
    the fence wins; None if there is none."""
    if not text:
        return None
    found = None
    for chunk in _candidates(text):
        try:
            g = json.loads(chunk)
        except (ValueError, RecursionError):
            continue
        if valid_grid(g):
            found = g
    return found


# --- statistics ------------------------------------------------------------------------------------------------------

def quantiles(xs, qs=(0.5, 0.9)):
    if not xs:
        return {f"p{int(q * 100)}": None for q in qs}
    s = sorted(xs)
    return {f"p{int(q * 100)}": s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))] for q in qs}


def km_quantiles(done, cens, qs=(0.5, 0.9)):
    """Kaplan-Meier quantiles of tokens-per-attempt with right-censoring (aborted/unfinished attempts are censored at
    the tokens they reached). None where the survival curve never drops that far."""
    events = sorted([(t, 1) for t in done] + [(t, 0) for t in cens])
    at_risk, surv, out, i = len(events), 1.0, {}, 0
    pending = list(qs)
    while i < len(events) and pending:
        t = events[i][0]
        d = c = 0
        while i < len(events) and events[i][0] == t:
            d += events[i][1]
            c += 1 - events[i][1]
            i += 1
        if d and at_risk:
            surv *= 1 - d / at_risk
        at_risk -= d + c
        while pending and 1 - surv >= pending[0]:
            out[f"km_p{int(pending.pop(0) * 100)}"] = t
    for q in pending:
        out[f"km_p{int(q * 100)}"] = None
    return out


def token_stats(rows):
    done = [r["completion_tokens"] for r in rows if r.get("state") == "done" and r.get("completion_tokens")]
    cens = [r["completion_tokens"] for r in rows if r.get("state") != "done" and r.get("completion_tokens")]
    out = {"n": len(rows), "n_done": len(done), "n_censored": len(cens),
           "mean_done": (sum(done) / len(done)) if done else None, "max_done": max(done) if done else None,
           "sum_all": sum(done) + sum(cens), "censored_over_40k": sum(1 for t in cens if t >= 40000),
           "censored_over_80k": sum(1 for t in cens if t >= 80000)}
    out.update(quantiles(done))
    out.update(km_quantiles(done, cens))
    return out


def arm_summary(rows):
    """rows: one effort arm. Accuracy is per test output; truncation = finish_reason 'length' among finished."""
    finished = [r for r in rows if r.get("state") == "done"]
    return {"tokens": token_stats(rows),
            "truncated": sum(1 for r in finished if r.get("finish_reason") == "length"),
            "parsed": sum(1 for r in finished if r.get("answer") is not None),
            "correct": sum(1 for r in finished if r.get("correct")),
            "finished": len(finished), "requests": len(rows)}


def pair_agreement(rows):
    """For an arm sampled twice per output: agreement count and its precision (agreeing answer correct)."""
    by = {}
    for r in rows:
        if r.get("state") == "done":
            by.setdefault(r["key"], []).append(r)
    agree = agree_ok = both = 0
    for rs in by.values():
        if len(rs) < 2:
            continue
        both += 1
        a, b = rs[0].get("answer"), rs[1].get("answer")
        if a is not None and a == b:
            agree += 1
            agree_ok += bool(rs[0].get("correct"))
    return {"pairs": both, "agree": agree, "agree_correct": agree_ok,
            "precision": (agree_ok / agree) if agree else None}
