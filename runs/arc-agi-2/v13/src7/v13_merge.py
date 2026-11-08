"""v13 fill rule (pre-registered, HANDOFF 2026-10-13): per test output, the answer of the verified programs (majority,
ties -> first verified) becomes attempt_1 and NVARC pass A's top-1 attempt_2; when the program answer already is one of
NVARC's two attempts the entry is left unchanged; when NVARC has no real grid, attempt_2 is the next distinct program
answer. Pure functions + atomic write; no third-party imports."""
import json
import os

PLACEHOLDER = [[0]]


def valid_grid(g):
    return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) for r in g)
            and 0 < len(g[0]) <= 30 and all(len(r) == len(g[0]) for r in g)
            and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))


def ranked_program_answers(answer_lists, i):
    """Distinct program answers for test output i, most-supported first, ties by first appearance."""
    order, count = [], {}
    for answers in answer_lists:
        try:
            g = answers[i]
        except (IndexError, TypeError):
            continue
        if not valid_grid(g):
            continue
        key = json.dumps(g)
        if key not in count:
            order.append(key)
            count[key] = 0
        count[key] += 1
    order.sort(key=lambda k: -count[k])  # stable: ties keep first appearance
    return [json.loads(k) for k in order]


def fill_entry(base_entry, programs):
    """base_entry: {'attempt_1', 'attempt_2'} from NVARC (may be placeholders); programs: ranked distinct grids."""
    if not programs:
        return dict(base_entry)
    nv = [base_entry.get(a) for a in ("attempt_1", "attempt_2")]
    nv = [g for g in nv if valid_grid(g) and g != PLACEHOLDER]
    best = programs[0]
    if best in nv:
        return dict(base_entry)
    second = nv[0] if nv else next((g for g in programs[1:] if g != best), None)
    if second is None:
        second = base_entry.get("attempt_1") if valid_grid(base_entry.get("attempt_1")) else PLACEHOLDER
    return {"attempt_1": best, "attempt_2": second}


def merge(test, base, results):
    """test: challenges dict; base: NVARC submission (same keys); results: task -> list of verified answer lists
    (one grid per test output, original frame). Returns (merged, n_changed_outputs)."""
    merged, changed = {}, 0
    for key, task in test.items():
        entries = []
        for i in range(len(task["test"])):
            try:
                b = dict(base[key][i])
            except Exception:  # noqa: BLE001  (missing/odd base entry: placeholders)
                b = {"attempt_1": PLACEHOLDER, "attempt_2": PLACEHOLDER}
            for a in ("attempt_1", "attempt_2"):
                if not valid_grid(b.get(a)):
                    b[a] = PLACEHOLDER
            e = fill_entry(b, ranked_program_answers(results.get(key, []), i))
            changed += e != b
            entries.append(e)
        merged[key] = entries
    return merged, changed


def validate(test, submission):
    if set(submission) != set(test):
        raise ValueError("submission keys differ from test keys")
    for k, task in test.items():
        entries = submission[k]
        if not isinstance(entries, list) or len(entries) != len(task["test"]):
            raise ValueError(f"{k}: wrong number of test entries")
        for e in entries:
            if not isinstance(e, dict) or set(e) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"{k}: attempts missing")
            if not all(valid_grid(g) for g in e.values()):
                raise ValueError(f"{k}: invalid grid")
    return True


def atomic_json(path, value):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f)
    os.replace(tmp, path)


def load_results(path):
    """task -> list of verified answer lists, from the driver's programs jsonl (verified rows only)."""
    res = {}
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("verified") and isinstance(r.get("answers"), list):
                    res.setdefault(r["task"], []).append(r["answers"])
    except OSError:
        pass
    return res


def finalize(work_dir="/kaggle/working", test_path=None):
    """Last notebook cell (rerun): rebuild submission.json from the saved NVARC base + verified programs; on any
    problem keep a valid submission (current file if valid, else the base). Never raises."""
    sub = os.path.join(work_dir, "submission.json")
    base_path = os.path.join(work_dir, "v13-base-submission.json")
    record = {"finalize": None}
    try:
        with open(test_path) as f:
            test = json.load(f)
        if not os.path.exists(base_path):
            record["finalize"] = "no base (LLM phase did not start); submission.json left as assembled"
            return record
        with open(base_path) as f:
            base = json.load(f)
        merged, changed = merge(test, base, load_results(os.path.join(work_dir, "v13-programs.jsonl")))
        validate(test, merged)
        atomic_json(sub, merged)
        record.update(finalize="merged", changed_outputs=changed)
    except Exception as e:  # noqa: BLE001
        record["finalize"] = f"error {type(e).__name__}: {e}"[:300]
        try:
            with open(test_path) as f:
                test = json.load(f)
            with open(sub) as f:
                validate(test, json.load(f))
            record["kept"] = "current submission.json (valid)"
        except Exception:  # noqa: BLE001
            try:
                with open(base_path) as f:
                    base = json.load(f)
                validate(test, base)
                atomic_json(sub, base)
                record["kept"] = "base"
            except Exception as e2:  # noqa: BLE001
                record["kept"] = f"nothing valid: {type(e2).__name__}"
    finally:
        print(json.dumps({"msg": "v13_finalize", **record}), flush=True)
    return record
