"""ARC-1/C1 train-pair invariants (prereg artifacts/verification/arc-agi-2/v13/prereg-arc1-c1-replay-v1.md).
An invariant is inferred only if it holds on every train pair; a candidate test output that violates an applicable
invariant is demoted (R: NVARC pool) or dropped (F: LLM program answers). Pure Python, no third-party imports.
Designed and soundness-checked on the training set + non-keys30 evaluation tasks only (prereg section 4)."""
from collections import Counter

SYMS = {
    "flip_lr": lambda g: [r[::-1] for r in g],
    "flip_ud": lambda g: g[::-1],
    "rot180": lambda g: [r[::-1] for r in g[::-1]],
    "transpose": lambda g: [list(r) for r in zip(*g)],
}


def shape(g):
    return (len(g), len(g[0]) if g else 0)


def colors(g):
    return {v for r in g for v in r}


def hist(g):
    return Counter(v for r in g for v in r)


def bg(g):
    return hist(g).most_common(1)[0][0]


def is_subgrid(small, big):
    (h, w), (H, W) = shape(small), shape(big)
    if h > H or w > W:
        return False
    return any(all(big[y + i][x:x + w] == small[i] for i in range(h))
               for y in range(H - h + 1) for x in range(W - w + 1))


def keeps_nonbg(i, o):
    b = bg(i)
    return shape(i) == shape(o) and all(o[y][x] == v for y, r in enumerate(i) for x, v in enumerate(r) if v != b)


def _ratio(a, b):
    """a == k*b (k int >= 1) -> ('mul', k); b == k*a -> ('div', k); else None."""
    if b and a % b == 0:
        return ("mul", a // b)
    if a and b % a == 0:
        return ("div", b // a)
    return None


def _apply_ratio(r, n):
    kind, k = r
    return n * k if kind == "mul" else (n // k if n % k == 0 else -1)


def infer(train):
    """train: list of (input_grid, output_grid). Returns {name: check(test_in, cand) -> bool}."""
    ins, outs = [p[0] for p in train], [p[1] for p in train]
    n = len(train)
    inv = {}
    if n == 0:
        return inv
    if all(shape(i) == shape(o) for i, o in train):
        inv["shape_same"] = lambda ti, c: shape(c) == shape(ti)
    if n >= 3 and len({shape(o) for o in outs}) == 1:
        k = shape(outs[0])
        inv["shape_const"] = lambda ti, c, k=k: shape(c) == k
    rh = {_ratio(shape(o)[0], shape(i)[0]) for i, o in train}
    rw = {_ratio(shape(o)[1], shape(i)[1]) for i, o in train}
    if len(rh) == 1 and len(rw) == 1 and None not in rh | rw and (rh | rw) != {("mul", 1)}:
        a, b = rh.pop(), rw.pop()
        inv["shape_ratio"] = lambda ti, c, a=a, b=b: shape(c) == (_apply_ratio(a, shape(ti)[0]), _apply_ratio(b, shape(ti)[1]))
    if all(shape(o) == shape(i)[::-1] for i, o in train) and not all(shape(i)[0] == shape(i)[1] for i in ins):
        inv["shape_transpose"] = lambda ti, c: shape(c) == shape(ti)[::-1]
    if all(colors(o) <= colors(i) for i, o in train):
        inv["colors_subset_in"] = lambda ti, c: colors(c) <= colors(ti)
    else:
        extra = set().union(*(colors(o) - colors(i) for i, o in train))
        if all(colors(o) - colors(i) == extra for i, o in train):  # the same new colors appear in every output
            inv["colors_subset_in_plus"] = lambda ti, c, e=frozenset(extra): colors(c) <= colors(ti) | e
    if all(colors(o) == colors(i) for i, o in train):
        inv["palette_eq"] = lambda ti, c: colors(c) == colors(ti)
    if all(shape(i) == shape(o) and hist(i) == hist(o) for i, o in train):
        inv["hist_eq"] = lambda ti, c: hist(c) == hist(ti)
    if all(keeps_nonbg(i, o) for i, o in train) and not all(i == o for i, o in train):
        inv["keep_nonbg"] = lambda ti, c: keeps_nonbg(ti, c)
    if n >= 3 and all(shape(o) != shape(i) and is_subgrid(o, i) for i, o in train):
        inv["subgrid_of_in"] = lambda ti, c: is_subgrid(c, ti)
    if n >= 3:
        for name, f in SYMS.items():
            if name == "transpose" and not all(shape(o)[0] == shape(o)[1] for o in outs):
                continue
            if all(f(o) == o for o in outs) and not all(f(i) == i for i in ins):
                inv["sym_" + name] = lambda ti, c, f=f: f(c) == c
    return inv


def violations(inv, test_in, cand, active=None):
    """Names of active invariants the candidate violates (active=None: all)."""
    out = []
    for name, chk in inv.items():
        if active is not None and name not in active:
            continue
        try:
            ok = chk(test_in, cand)
        except Exception:  # noqa: BLE001  (malformed candidate: treat as violation)
            ok = False
        if not ok:
            out.append(name)
    return out


def stable_filter(ranked, inv, test_in, active=None, keep=2):
    """R: candidates passing every active invariant keep their order and go first; violators refill the tail."""
    good = [g for g in ranked if not violations(inv, test_in, g, active)]
    bad = [g for g in ranked if violations(inv, test_in, g, active)]
    return (good + bad)[:keep] if keep else good + bad
