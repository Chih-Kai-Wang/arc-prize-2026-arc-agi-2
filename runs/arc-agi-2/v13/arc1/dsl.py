"""Clean-room CPU DSL solver for ARC grids (ARC-1 "D", v1).

Implemented strictly from `runs/arc-agi-2/v13/arc1/dsl-spec-v1.md`:
- Self-contained Python 3.11 module `dsl.py` using Python stdlib and numpy only.
- Grid format (`runs/arc-agi-2/v13/arc1/dsl-spec-v1.md`, lines 7-8, 18):
  rectangular list of lists of ints `0..9`, dimensions `1x1` up to `30x30` (`1..30` per side),
  `2-10` train pairs and `1-3` test input grids, default colour `0` usually background.
- Exact API (`runs/arc-agi-2/v13/arc1/dsl-spec-v1.md`, lines 12-17):
  `solve(train_pairs, test_inputs, time_limit=10.0)` returning `list[(program_name, test_outputs)]`.
- Search (`runs/arc-agi-2/v13/arc1/dsl-spec-v1.md`, lines 21-31):
  BFS over compositions of up to `3` unary primitives (depths `1`, `2`, `3`), deduplicating
  intermediate train results, testing learned pixelwise colour maps `T+m`, learning/enumerating
  parameters over colours `0..9` and scale factors `2..4`, with early shape rejection.
- Primitives (`runs/arc-agi-2/v13/arc1/dsl-spec-v1.md`, lines 33-53):
  all `10` specified primitive categories implemented, total code kept under `~900` lines (`line 57`).
- Verification note: all requirements come solely from `runs/arc-agi-2/v13/arc1/dsl-spec-v1.md`
  (61 lines, 4397 bytes); no external task files or runtime test suites exist in the workspace to
  verify against beyond the self-test specified on line 58.
"""

from collections import Counter, deque
import time
from typing import Callable
import numpy as np

# Per-call caches cleared in solve() finally block to maintain zero state between calls.
_OBJ_CACHE: dict[tuple, list[dict]] = {}
_SPLIT_CACHE: dict[tuple, tuple | None] = {}
_ENC_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def is_valid_grid(g) -> bool:
    """Check that g is a 2D int numpy array with shape in 1..30 and values in 0..9."""
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return False
    h, w = g.shape
    if h < 1 or h > 30 or w < 1 or w > 30:
        return False
    if g.dtype.kind not in ("i", "u"):
        return False
    mn, mx = int(g.min()), int(g.max())
    return 0 <= mn and mx <= 9


def to_grid_array(g) -> np.ndarray | None:
    """Convert a list-of-lists grid to an int8 numpy array, or None if invalid."""
    if not isinstance(g, (list, tuple)) or not g:
        return None
    w = None
    for row in g:
        if not isinstance(row, (list, tuple)) or not row:
            return None
        if w is None:
            w = len(row)
        elif len(row) != w:
            return None
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 or v > 9:
                return None
    arr = np.array(g, dtype=np.int8)
    return arr if is_valid_grid(arr) else None


def to_list_grid(arr: np.ndarray) -> list[list[int]]:
    """Convert a 2D numpy array back to a list of lists of Python ints."""
    return [[int(x) for x in row] for row in arr.tolist()]


def infer_bg(g: np.ndarray) -> int:
    """Return 0 usually, or the dominant background colour when 0 is absent/minor."""
    vals, cnts = np.unique(g, return_counts=True)
    if len(vals) == 0:
        return 0
    top_c = int(vals[np.argmax(cnts)])
    top_cnt = int(cnts.max())
    zero_cnt = int(cnts[vals == 0][0]) if 0 in vals else 0
    if zero_cnt == 0:
        return top_c
    if top_c != 0 and top_cnt > 0.55 * g.size and zero_cnt < 0.15 * g.size:
        return top_c
    return 0


def find_objects(
    g: np.ndarray, conn: int = 4, multicolor: bool = False, bg: int | None = 0
) -> list[dict]:
    """Extract connected non-background objects with bounding boxes and masks."""
    actual_bg = infer_bg(g) if bg is None else bg
    key = (g.shape, g.tobytes(), conn, multicolor, actual_bg)
    cached = _OBJ_CACHE.get(key)
    if cached is not None:
        return cached

    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    visited[g == actual_bg] = True
    dirs = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if conn == 4
        else ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    )
    objs: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            c_val = int(g[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            pixels = [(r, c)]
            r0 = r1 = r
            c0 = c1 = c
            while q:
                cr, cc = q.popleft()
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if multicolor or int(g[nr, nc]) == c_val:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            pixels.append((nr, nc))
                            if nr < r0:
                                r0 = nr
                            elif nr > r1:
                                r1 = nr
                            if nc < c0:
                                c0 = nc
                            elif nc > c1:
                                c1 = nc

            bh, bw = r1 - r0 + 1, c1 - c0 + 1
            norm_mask = np.zeros((bh, bw), dtype=bool)
            subgrid = np.full((bh, bw), actual_bg, dtype=np.int8)
            colors = []
            for pr, pc in pixels:
                val = int(g[pr, pc])
                norm_mask[pr - r0, pc - c0] = True
                subgrid[pr - r0, pc - c0] = val
                colors.append(val)
            obj_color = c_val if not multicolor else Counter(colors).most_common(1)[0][0]
            objs.append(
                {
                    "pixels": pixels,
                    "size": len(pixels),
                    "color": obj_color,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "norm_mask": norm_mask,
                    "subgrid": subgrid,
                    "raw_crop": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                }
            )

    _OBJ_CACHE[key] = objs
    return objs


def enclosed_masks(g: np.ndarray, conn: int = 4, bg: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (border_reachable_bg_mask, enclosed_bg_mask)."""
    key = (g.shape, g.tobytes(), conn, bg)
    cached = _ENC_CACHE.get(key)
    if cached is not None:
        return cached

    h, w = g.shape
    bg_mask = g == bg
    reach = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not reach[r, c]:
                reach[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not reach[r, c]:
                reach[r, c] = True
                q.append((r, c))

    dirs = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if conn == 4
        else ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    )
    while q:
        cr, cc = q.popleft()
        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))

    enc = bg_mask & ~reach
    _ENC_CACHE[key] = (reach, enc)
    return reach, enc


def _spans_from_sep(n: int, sep_indices: set[int]) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i in range(n):
        if i in sep_indices:
            if start is not None:
                spans.append((start, i))
                start = None
        else:
            if start is None:
                start = i
    if start is not None:
        spans.append((start, n))
    return spans


def split_grid_cells(g: np.ndarray) -> tuple[list[list[np.ndarray]], list[np.ndarray], int] | None:
    """Detect separator rows/cols of one colour and split grid into cells."""
    key = (g.shape, g.tobytes())
    if key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]

    h, w = g.shape
    bg = infer_bg(g)
    row_unif = [int(g[r, 0]) if np.all(g[r, :] == g[r, 0]) else -1 for r in range(h)]
    col_unif = [int(g[0, c]) if np.all(g[:, c] == g[0, c]) else -1 for c in range(w)]

    candidate_colors = []
    for c in set(row_unif + col_unif):
        if c >= 0 and c != bg:
            candidate_colors.append(c)
    if bg in row_unif or bg in col_unif:
        candidate_colors.append(bg)

    for sep_c in candidate_colors:
        r_seps = {r for r, c in enumerate(row_unif) if c == sep_c}
        c_seps = {c for c, col_c in enumerate(col_unif) if col_c == sep_c}
        r_spans = _spans_from_sep(h, r_seps)
        c_spans = _spans_from_sep(w, c_seps)
        if not r_spans or not c_spans or len(r_spans) * len(c_spans) < 2:
            continue
        if sep_c == bg:
            shapes = {(r1 - r0, c1 - c0) for r0, r1 in r_spans for c0, c1 in c_spans}
            if len(shapes) != 1:
                continue
        cells_2d = [[g[r0:r1, c0:c1].copy() for c0, c1 in c_spans] for r0, r1 in r_spans]
        flat = [cell for row in cells_2d for cell in row]
        res = (cells_2d, flat, sep_c)
        _SPLIT_CACHE[key] = res
        return res

    _SPLIT_CACHE[key] = None
    return None


# =============================================================================
# 1. Geometry Primitives
# =============================================================================

def p_identity(g: np.ndarray) -> np.ndarray:
    return g.copy()


def p_rot90(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, 1).copy()


def p_rot180(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, 2).copy()


def p_rot270(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, 3).copy()


def p_flip_lr(g: np.ndarray) -> np.ndarray:
    return np.fliplr(g).copy()


def p_flip_ud(g: np.ndarray) -> np.ndarray:
    return np.flipud(g).copy()


def p_transpose(g: np.ndarray) -> np.ndarray:
    return g.T.copy()


def p_anti_transpose(g: np.ndarray) -> np.ndarray:
    return np.fliplr(np.flipud(g.T)).copy()


def p_crop_border1(g: np.ndarray) -> np.ndarray | None:
    if g.shape[0] <= 2 or g.shape[1] <= 2:
        return None
    return g[1:-1, 1:-1].copy()


def p_left_half(g: np.ndarray) -> np.ndarray | None:
    if g.shape[1] < 2:
        return None
    return g[:, : g.shape[1] // 2].copy()


def p_right_half(g: np.ndarray) -> np.ndarray | None:
    if g.shape[1] < 2:
        return None
    return g[:, (g.shape[1] + 1) // 2 :].copy()


def p_top_half(g: np.ndarray) -> np.ndarray | None:
    if g.shape[0] < 2:
        return None
    return g[: g.shape[0] // 2, :].copy()


def p_bottom_half(g: np.ndarray) -> np.ndarray | None:
    if g.shape[0] < 2:
        return None
    return g[(g.shape[0] + 1) // 2 :, :].copy()


# =============================================================================
# 2. Cropping Primitives
# =============================================================================

def p_crop_non_bg(g: np.ndarray, bg: int | None = 0) -> np.ndarray | None:
    actual_bg = infer_bg(g) if bg is None else bg
    ys, xs = np.where(g != actual_bg)
    if len(ys) == 0:
        return None
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()


def _crop_obj_by_size(
    g: np.ndarray, largest: bool, conn: int, multicolor: bool, masked: bool = False, bg: int | None = 0
) -> np.ndarray | None:
    objs = find_objects(g, conn=conn, multicolor=multicolor, bg=bg)
    if not objs:
        return None
    key_fn = lambda o: (o["size"], (o["r1"] - o["r0"] + 1) * (o["c1"] - o["c0"] + 1), -o["r0"], -o["c0"])
    chosen = max(objs, key=key_fn) if largest else min(objs, key=key_fn)
    return chosen["subgrid"].copy() if masked else chosen["raw_crop"].copy()


def p_crop_second_largest(g: np.ndarray) -> np.ndarray | None:
    objs = find_objects(g, conn=4, multicolor=False, bg=0)
    if len(objs) < 2:
        return None
    objs_sorted = sorted(objs, key=lambda o: (-o["size"], o["r0"], o["c0"]))
    return objs_sorted[1]["raw_crop"].copy()


def p_crop_unique_color_obj(g: np.ndarray, conn: int = 4, masked: bool = False) -> np.ndarray | None:
    objs = find_objects(g, conn=conn, multicolor=False, bg=0)
    if not objs:
        return None
    counts = Counter(o["color"] for o in objs)
    uniq = [o for o in objs if counts[o["color"]] == 1]
    if not uniq:
        return None
    chosen = min(uniq, key=lambda o: (int(np.sum(g == o["color"])), o["r0"], o["c0"]))
    return chosen["subgrid"].copy() if masked else chosen["raw_crop"].copy()


def _shape_sig(mask: np.ndarray, rot_inv: bool = False) -> tuple:
    if not rot_inv:
        return (mask.shape, mask.tobytes())
    sigs = []
    for k in range(4):
        r = np.rot90(mask, k)
        sigs.append((r.shape, r.tobytes()))
        f = np.fliplr(r)
        sigs.append((f.shape, f.tobytes()))
    return min(sigs)


def p_crop_unique_shape_obj(
    g: np.ndarray, conn: int = 4, multicolor: bool = False, rot_inv: bool = False
) -> np.ndarray | None:
    objs = find_objects(g, conn=conn, multicolor=multicolor, bg=0)
    if len(objs) < 2:
        return None
    sigs = [_shape_sig(o["norm_mask"], rot_inv=rot_inv) for o in objs]
    counts = Counter(sigs)
    for o, s in zip(objs, sigs):
        if counts[s] == 1:
            return o["raw_crop"].copy()
    return None


def p_crop_color_bbox(g: np.ndarray, c: int) -> np.ndarray | None:
    ys, xs = np.where(g == c)
    if len(ys) == 0:
        return None
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()


def p_crop_inside_color_bbox(g: np.ndarray, c: int) -> np.ndarray | None:
    ys, xs = np.where(g == c)
    if len(ys) == 0:
        return None
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    return g[r0 + 1 : r1, c0 + 1 : c1].copy()


# =============================================================================
# 3. Scaling and Tiling Primitives
# =============================================================================

def p_upscale(g: np.ndarray, ky: int, kx: int) -> np.ndarray | None:
    h, w = g.shape
    if h * ky > 30 or w * kx > 30 or ky < 1 or kx < 1:
        return None
    return np.repeat(np.repeat(g, ky, axis=0), kx, axis=1)


def p_fractal_upscale(g: np.ndarray, recolor: bool = False) -> np.ndarray | None:
    h, w = g.shape
    if h * h > 30 or w * w > 30:
        return None
    bg = infer_bg(g)
    out = np.full((h * h, w * w), bg, dtype=np.int8)
    mask = g != bg
    for r in range(h):
        for c in range(w):
            if g[r, c] != bg:
                if recolor:
                    blk = np.where(mask, g[r, c], bg).astype(np.int8)
                    out[r * h : (r + 1) * h, c * w : (c + 1) * w] = blk
                else:
                    out[r * h : (r + 1) * h, c * w : (c + 1) * w] = g
    return out


def p_downscale_topleft(g: np.ndarray, ky: int, kx: int) -> np.ndarray | None:
    h, w = g.shape
    if ky < 1 or kx < 1 or h % ky != 0 or w % kx != 0:
        return None
    return g[::ky, ::kx].copy()


def p_downscale_majority(
    g: np.ndarray, ky: int, kx: int, prefer_nonbg: bool = False
) -> np.ndarray | None:
    h, w = g.shape
    if ky < 1 or kx < 1 or h % ky != 0 or w % kx != 0:
        return None
    oh, ow = h // ky, w // kx
    out = np.zeros((oh, ow), dtype=np.int8)
    for r in range(oh):
        for c in range(ow):
            blk = g[r * ky : (r + 1) * ky, c * kx : (c + 1) * kx].ravel()
            if prefer_nonbg:
                nb = blk[blk != 0]
                if len(nb) > 0:
                    blk = nb
            vals, cnts = np.unique(blk, return_counts=True)
            out[r, c] = vals[np.argmax(cnts)]
    return out


def p_compress_unique_lines(g: np.ndarray) -> np.ndarray | None:
    h, w = g.shape
    keep_r = [0] + [r for r in range(1, h) if not np.array_equal(g[r], g[r - 1])]
    out = g[keep_r, :]
    keep_c = [0] + [c for c in range(1, w) if not np.array_equal(out[:, c], out[:, c - 1])]
    out = out[:, keep_c].copy()
    return None if out.shape == g.shape else out


def p_tile(g: np.ndarray, ny: int, nx: int) -> np.ndarray | None:
    if g.shape[0] * ny > 30 or g.shape[1] * nx > 30:
        return None
    return np.tile(g, (ny, nx))


def p_mirror_tile_2x2(g: np.ndarray, mode: int = 0) -> np.ndarray | None:
    if g.shape[0] * 2 > 30 or g.shape[1] * 2 > 30:
        return None
    lr = np.fliplr(g)
    ud = np.flipud(g)
    r180 = np.rot90(g, 2)
    if mode == 0:
        return np.vstack([np.hstack([g, lr]), np.hstack([ud, r180])])
    if mode == 1:
        return np.vstack([np.hstack([r180, ud]), np.hstack([lr, g])])
    if mode == 2:
        return np.vstack([np.hstack([lr, g]), np.hstack([r180, ud])])
    return np.vstack([np.hstack([ud, r180]), np.hstack([g, lr])])


def p_hconcat_fliplr(g: np.ndarray, rev: bool = False) -> np.ndarray | None:
    if g.shape[1] * 2 > 30:
        return None
    lr = np.fliplr(g)
    return np.hstack([lr, g]) if rev else np.hstack([g, lr])


def p_vconcat_flipud(g: np.ndarray, rev: bool = False) -> np.ndarray | None:
    if g.shape[0] * 2 > 30:
        return None
    ud = np.flipud(g)
    return np.vstack([ud, g]) if rev else np.vstack([g, ud])


# =============================================================================
# 4. Separator Lines and Cell Combination Primitives
# =============================================================================

def _combine_masks(masks: list[np.ndarray], op: str) -> np.ndarray | None:
    if len(masks) < 2:
        return None
    if op == "and":
        return np.logical_and.reduce(masks)
    if op == "or":
        return np.logical_or.reduce(masks)
    if op == "xor":
        return np.bitwise_xor.reduce(masks)
    if op == "uniq1":
        return sum(m.astype(int) for m in masks) == 1
    if op == "nor":
        return ~np.logical_or.reduce(masks)
    if op == "diff":
        return masks[0] & ~np.logical_or.reduce(masks[1:])
    if op == "rdiff":
        return masks[-1] & ~masks[0]
    if op == "maj":
        return sum(m.astype(int) for m in masks) > (len(masks) // 2)
    return None


def p_cells_op(g: np.ndarray, op: str, out_c: int, fallback_halves: bool = False) -> np.ndarray | None:
    sp = split_grid_cells(g)
    cells = None
    if sp is not None:
        _, flat, _ = sp
        if len({c.shape for c in flat}) == 1:
            cells = flat
    elif fallback_halves:
        h, w = g.shape
        if w >= 2 and w % 2 == 0:
            cells = [g[:, : w // 2], g[:, w // 2 :]]
        elif h >= 2 and h % 2 == 0:
            cells = [g[: h // 2, :], g[h // 2 :, :]]
    if not cells or len(cells) < 2:
        return None
    masks = [c != 0 for c in cells]
    res_mask = _combine_masks(masks, op)
    if res_mask is None:
        return None
    return np.where(res_mask, out_c, 0).astype(np.int8)


def p_cells_overlay(g: np.ndarray, reverse: bool = False) -> np.ndarray | None:
    sp = split_grid_cells(g)
    if sp is None:
        return None
    _, flat, _ = sp
    if len({c.shape for c in flat}) != 1:
        return None
    seq = reversed(flat) if reverse else flat
    out = np.zeros_like(flat[0])
    for cell in seq:
        m = out == 0
        out[m] = cell[m]
    return out


def p_pick_unique_cell(g: np.ndarray) -> np.ndarray | None:
    sp = split_grid_cells(g)
    if sp is None:
        return None
    _, flat, _ = sp
    if len(flat) < 3:
        return None
    for key_fn in (
        lambda c: (c.shape, c.tobytes()),
        lambda c: ((c != 0).shape, (c != 0).tobytes()),
        lambda c: int(np.sum(c != 0)),
        lambda c: tuple(sorted( set(int(x) for x in c.ravel() if x != 0) )),
    ):
        keys = [key_fn(c) for c in flat]
        cnt = Counter(keys)
        for c, k in zip(flat, keys):
            if cnt[k] == 1:
                return c.copy()
    return None


def p_pick_extreme_nonbg_cell(g: np.ndarray, most: bool = True) -> np.ndarray | None:
    sp = split_grid_cells(g)
    if sp is None:
        return None
    _, flat, _ = sp
    if not most:
        non_empty = [c for c in flat if np.any(c != 0)]
        if non_empty:
            flat = non_empty
    counts = [int(np.sum(c != 0)) for c in flat]
    idx = int(np.argmax(counts)) if most else int(np.argmin(counts))
    return flat[idx].copy()


def p_condense_cells(g: np.ndarray) -> np.ndarray | None:
    sp = split_grid_cells(g)
    if sp is None:
        return None
    cells_2d, _, _ = sp
    rh, cw = len(cells_2d), len(cells_2d[0])
    if rh > 30 or cw > 30:
        return None
    out = np.zeros((rh, cw), dtype=np.int8)
    for r in range(rh):
        for c in range(cw):
            blk = cells_2d[r][c].ravel()
            nb = blk[blk != 0]
            if len(nb) > 0:
                vals, cnts = np.unique(nb, return_counts=True)
                out[r, c] = vals[np.argmax(cnts)]
    return out


def p_remove_separators(g: np.ndarray) -> np.ndarray | None:
    sp = split_grid_cells(g)
    if sp is None:
        return None
    cells_2d, _, _ = sp
    try:
        rows = [np.hstack(row) for row in cells_2d]
        return np.vstack(rows).astype(np.int8)
    except ValueError:
        return None


# =============================================================================
# 5. Physics and Fill Primitives
# =============================================================================

def p_gravity(g: np.ndarray, direction: str) -> np.ndarray:
    bg = infer_bg(g)
    h, w = g.shape
    out = np.full((h, w), bg, dtype=np.int8)
    if direction in ("down", "up"):
        for c in range(w):
            col = g[:, c]
            vals = col[col != bg]
            n = len(vals)
            if n:
                if direction == "down":
                    out[h - n :, c] = vals
                else:
                    out[:n, c] = vals
    else:
        for r in range(h):
            row = g[r, :]
            vals = row[row != bg]
            n = len(vals)
            if n:
                if direction == "right":
                    out[r, w - n :] = vals
                else:
                    out[r, :n] = vals
    return out


def p_fill_enclosed(g: np.ndarray, c: int, conn: int = 4) -> np.ndarray | None:
    _, enc = enclosed_masks(g, conn=conn, bg=0)
    if not enc.any():
        return None
    out = g.copy()
    out[enc] = c
    return out


def p_fill_enclosed_by_neighbor(g: np.ndarray) -> np.ndarray | None:
    _, enc = enclosed_masks(g, conn=4, bg=0)
    if not enc.any():
        return None
    h, w = g.shape
    out = g.copy()
    visited = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if enc[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                comp = [(r, c)]
                nbr_colors = []
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if enc[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                                comp.append((nr, nc))
                            elif g[nr, nc] != 0:
                                nbr_colors.append(int(g[nr, nc]))
                if nbr_colors:
                    fill_c = Counter(nbr_colors).most_common(1)[0][0]
                    for pr, pc in comp:
                        out[pr, pc] = fill_c
    return out


def p_flood_borders(g: np.ndarray, c: int, conn: int = 4) -> np.ndarray | None:
    reach, _ = enclosed_masks(g, conn=conn, bg=0)
    if not reach.any():
        return None
    out = g.copy()
    out[reach] = c
    return out


def p_draw_lines_through_color(g: np.ndarray, c: int, axis: str) -> np.ndarray | None:
    ys, xs = np.where(g == c)
    if len(ys) == 0:
        return None
    out = g.copy()
    if axis in ("h", "both"):
        for r in np.unique(ys):
            m = out[r, :] == 0
            out[r, m] = c
    if axis in ("v", "both"):
        for col in np.unique(xs):
            m = out[:, col] == 0
            out[m, col] = c
    return out


def p_connect_aligned_same_color(g: np.ndarray, axis: str = "both") -> np.ndarray | None:
    h, w = g.shape
    out = g.copy()
    changed = False
    if axis in ("h", "both"):
        for r in range(h):
            row = g[r, :]
            for c in np.unique(row):
                if c == 0:
                    continue
                xs = np.where(row == c)[0]
                if len(xs) >= 2:
                    seg = out[r, xs[0] : xs[-1] + 1]
                    m = seg == 0
                    if m.any():
                        seg[m] = c
                        changed = True
    if axis in ("v", "both"):
        for col_i in range(w):
            col = g[:, col_i]
            for c in np.unique(col):
                if c == 0:
                    continue
                ys = np.where(col == c)[0]
                if len(ys) >= 2:
                    seg = out[ys[0] : ys[-1] + 1, col_i]
                    m = seg == 0
                    if m.any():
                        seg[m] = c
                        changed = True
    return out if changed else None


def p_extend_pixels(
    g: np.ndarray, direction: str, only_dots: bool = False, must_hit: bool = False
) -> np.ndarray | None:
    h, w = g.shape
    out = g.copy()
    dir_map = {"down": [(1, 0)], "up": [(-1, 0)], "right": [(0, 1)], "left": [(0, -1)]}
    dirs = dir_map.get(direction, [(1, 0), (-1, 0), (0, 1), (0, -1)])

    if only_dots:
        objs = find_objects(g, conn=4, multicolor=False, bg=0)
        sources = [o["pixels"][0] for o in objs if o["size"] == 1]
    else:
        ys, xs = np.where(g != 0)
        sources = list(zip(ys.tolist(), xs.tolist()))

    if not sources:
        return None
    changed = False
    for r, c in sources:
        val = int(g[r, c])
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w and g[nr, nc] == 0):
                continue
            path = []
            while 0 <= nr < h and 0 <= nc < w and g[nr, nc] == 0:
                path.append((nr, nc))
                nr += dr
                nc += dc
            hit = 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 0
            if must_hit and not hit:
                continue
            for pr, pc in path:
                out[pr, pc] = val
                changed = True
    return out if changed else None


# =============================================================================
# 6. Symmetry Completion Primitives
# =============================================================================

def p_sym_complete(g: np.ndarray, mode: str, bbox_only: bool = False) -> np.ndarray | None:
    bg = infer_bg(g)
    if bbox_only:
        ys, xs = np.where(g != bg)
        if len(ys) == 0:
            return None
        r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        sub = p_sym_complete(g[r0 : r1 + 1, c0 : c1 + 1], mode, bbox_only=False)
        if sub is None:
            return None
        out = g.copy()
        out[r0 : r1 + 1, c0 : c1 + 1] = sub
        return out

    h, w = g.shape
    out = g.copy()
    partners = []
    if mode == "lr":
        partners = [np.fliplr(g)]
    elif mode == "ud":
        partners = [np.flipud(g)]
    elif mode == "rot180":
        partners = [np.rot90(g, 2)]
    elif mode == "both":
        partners = [np.fliplr(g), np.flipud(g), np.rot90(g, 2)]
    elif mode == "rot90":
        if h != w:
            return None
        partners = [np.rot90(g, 1), np.rot90(g, 2), np.rot90(g, 3)]
    elif mode == "transpose":
        if h != w:
            return None
        partners = [g.T]
    elif mode == "antitranspose":
        if h != w:
            return None
        partners = [np.fliplr(np.flipud(g.T))]

    for p in partners:
        m = out == bg
        out[m] = p[m]
    return out


def p_sym_fill_hole(g: np.ndarray, hole_c: int, mode: str) -> np.ndarray | None:
    if not np.any(g == hole_c):
        return None
    h, w = g.shape
    out = g.copy()
    partners = []
    if mode == "lr":
        partners = [np.fliplr(g)]
    elif mode == "ud":
        partners = [np.flipud(g)]
    elif mode == "rot180":
        partners = [np.rot90(g, 2)]
    elif mode == "both":
        partners = [np.fliplr(g), np.flipud(g), np.rot90(g, 2)]
    elif mode == "rot90":
        if h != w:
            return None
        partners = [np.rot90(g, 1), np.rot90(g, 2), np.rot90(g, 3)]
    for p in partners:
        m = (out == hole_c) & (p != hole_c)
        out[m] = p[m]
    return out


# =============================================================================
# 7. Colour Operations
# =============================================================================

def p_keep_color(g: np.ndarray, c: int) -> np.ndarray | None:
    if not np.any(g == c):
        return None
    return np.where(g == c, g, 0).astype(np.int8)


def p_remove_color(g: np.ndarray, c: int) -> np.ndarray | None:
    if not np.any(g == c):
        return None
    return np.where(g == c, 0, g).astype(np.int8)


def p_swap_colors_by_freq(g: np.ndarray, top2: bool = True) -> np.ndarray | None:
    bg = infer_bg(g)
    nb = g[g != bg]
    if len(nb) == 0:
        return None
    vals, cnts = np.unique(nb, return_counts=True)
    if len(vals) < 2:
        return None
    order = sorted(range(len(vals)), key=lambda i: (-int(cnts[i]), int(vals[i])))
    c1 = int(vals[order[0]])
    c2 = int(vals[order[1] if top2 else order[-1]])
    out = g.copy()
    out[g == c1] = c2
    out[g == c2] = c1
    return out


def p_recolor_objects_canonical(
    g: np.ndarray, mode: str, conn: int = 4, multicolor: bool = False
) -> np.ndarray | None:
    objs = find_objects(g, conn=conn, multicolor=multicolor, bg=0)
    if not objs:
        return None
    out = np.zeros_like(g)
    if mode == "size_clamp":
        for o in objs:
            c = min(9, max(1, o["size"]))
            for r, col in o["pixels"]:
                out[r, col] = c
    else:
        sizes = sorted({o["size"] for o in objs}, reverse=(mode == "rank_desc"))
        s_to_c = {s: min(9, idx + 1) for idx, s in enumerate(sizes)}
        for o in objs:
            c = s_to_c[o["size"]]
            for r, col in o["pixels"]:
                out[r, col] = c
    return out


# =============================================================================
# 8. Denoise Primitives
# =============================================================================

def p_remove_isolated_pixels(g: np.ndarray, conn: int = 4, same_color: bool = True) -> np.ndarray:
    h, w = g.shape
    bg = infer_bg(g)
    out = g.copy()
    dirs = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if conn == 4
        else ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    )
    for r in range(h):
        for c in range(w):
            val = int(g[r, c])
            if val == bg:
                continue
            has_nbr = False
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if (same_color and int(g[nr, nc]) == val) or (not same_color and int(g[nr, nc]) != bg):
                        has_nbr = True
                        break
            if not has_nbr:
                out[r, c] = bg
    return out


def p_remove_small_objects(
    g: np.ndarray, min_size: int, conn: int = 4, multicolor: bool = False
) -> np.ndarray | None:
    objs = find_objects(g, conn=conn, multicolor=multicolor, bg=0)
    if not objs:
        return None
    out = g.copy()
    changed = False
    for o in objs:
        if o["size"] < min_size:
            for r, c in o["pixels"]:
                out[r, c] = 0
            changed = True
    return out if changed else None


def p_keep_extreme_object_in_place(
    g: np.ndarray, largest: bool = True, conn: int = 4, multicolor: bool = False
) -> np.ndarray | None:
    objs = find_objects(g, conn=conn, multicolor=multicolor, bg=0)
    if len(objs) < 2:
        return None
    target_sz = max(o["size"] for o in objs) if largest else min(o["size"] for o in objs)
    out = np.zeros_like(g)
    for o in objs:
        if o["size"] == target_sz:
            for r, c in o["pixels"]:
                out[r, c] = g[r, c]
    return out


# =============================================================================
# 9. Counting Outputs
# =============================================================================

def p_extreme_freq_color_1x1(g: np.ndarray, most: bool = True) -> np.ndarray | None:
    bg = infer_bg(g)
    nb = g[g != bg]
    if len(nb) == 0:
        return None
    vals, cnts = np.unique(nb, return_counts=True)
    order = sorted(range(len(vals)), key=lambda i: (-int(cnts[i]), int(vals[i])))
    c = int(vals[order[0] if most else order[-1]])
    return np.array([[c]], dtype=np.int8)


def p_count_objects_bar(
    g: np.ndarray, c: int, horizontal: bool = True, conn: int = 4, multicolor: bool = False
) -> np.ndarray | None:
    n = len(find_objects(g, conn=conn, multicolor=multicolor, bg=0))
    if n < 1 or n > 30:
        return None
    shape = (1, n) if horizontal else (n, 1)
    return np.full(shape, c, dtype=np.int8)


# =============================================================================
# 10. Outline and Rectangle Interior Primitives
# =============================================================================

def p_outline_objects_internal(g: np.ndarray, c: int, hollow: bool = True) -> np.ndarray | None:
    objs = find_objects(g, conn=4, multicolor=False, bg=0)
    if not objs:
        return None
    h, w = g.shape
    out = np.zeros_like(g) if hollow else g.copy()
    for o in objs:
        pset = set(o["pixels"])
        for r, col in o["pixels"]:
            is_inner = (
                r > 0
                and r < h - 1
                and col > 0
                and col < w - 1
                and (r - 1, col) in pset
                and (r + 1, col) in pset
                and (r, col - 1) in pset
                and (r, col + 1) in pset
            )
            if not is_inner:
                out[r, col] = o["color"] if c < 0 else c
            elif not hollow:
                out[r, col] = g[r, col]
    return out


def p_outline_objects_external(g: np.ndarray, c: int, conn: int = 4) -> np.ndarray | None:
    h, w = g.shape
    non_bg = g != 0
    if not non_bg.any():
        return None
    dirs = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if conn == 4
        else ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    )
    out = g.copy()
    changed = False
    for r in range(h):
        for col in range(w):
            if g[r, col] == 0:
                for dr, dc in dirs:
                    nr, nc = r + dr, col + dc
                    if 0 <= nr < h and 0 <= nc < w and non_bg[nr, nc]:
                        out[r, col] = c
                        changed = True
                        break
    return out if changed else None


def p_fill_rectangles_interior(g: np.ndarray, c: int | None = None) -> np.ndarray | None:
    objs = find_objects(g, conn=4, multicolor=False, bg=0)
    if not objs:
        return None
    out = g.copy()
    changed = False
    for o in objs:
        r0, r1, c0, c1 = o["r0"], o["r1"], o["c0"], o["c1"]
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            sub = out[r0 + 1 : r1, c0 + 1 : c1]
            m = sub == 0
            if m.any():
                sub[m] = o["color"] if c is None else c
                changed = True
    return out if changed else None


def p_fill_object_bboxes(g: np.ndarray, conn: int = 4, multicolor: bool = False) -> np.ndarray | None:
    objs = find_objects(g, conn=conn, multicolor=multicolor, bg=0)
    if not objs:
        return None
    out = g.copy()
    for o in objs:
        out[o["r0"] : o["r1"] + 1, o["c0"] : o["c1"] + 1] = o["color"]
    return out


# =============================================================================
# Parametric Learning and Instantiation
# =============================================================================

def _build_learned_recolor_prims(
    train_in: list[np.ndarray], train_out: list[np.ndarray]
) -> list[tuple[str, Callable, str]]:
    if not all(i.shape == o.shape for i, o in zip(train_in, train_out)):
        return []
    prims: list[tuple[str, Callable, str]] = []
    for conn in (4, 8):
        for multi in (False, True):
            size_map: dict[int, int] = {}
            rank_desc_map: dict[int, int] = {}
            rank_asc_map: dict[int, int] = {}
            ok_size = ok_rdesc = ok_rasc = True

            for inp, out in zip(train_in, train_out):
                objs = find_objects(inp, conn=conn, multicolor=multi, bg=0)
                if not objs:
                    ok_size = ok_rdesc = ok_rasc = False
                    break
                sizes_desc = sorted({o["size"] for o in objs}, reverse=True)
                sizes_asc = list(reversed(sizes_desc))
                rdesc_idx = {s: idx for idx, s in enumerate(sizes_desc)}
                rasc_idx = {s: idx for idx, s in enumerate(sizes_asc)}

                for o in objs:
                    out_vals = {int(out[r, c]) for r, c in o["pixels"]}
                    if len(out_vals) != 1:
                        ok_size = ok_rdesc = ok_rasc = False
                        break
                    target_c = next(iter(out_vals))
                    sz = o["size"]
                    if sz in size_map and size_map[sz] != target_c:
                        ok_size = False
                    size_map[sz] = target_c

                    rd = rdesc_idx[sz]
                    if rd in rank_desc_map and rank_desc_map[rd] != target_c:
                        ok_rdesc = False
                    rank_desc_map[rd] = target_c

                    ra = rasc_idx[sz]
                    if ra in rank_asc_map and rank_asc_map[ra] != target_c:
                        ok_rasc = False
                    rank_asc_map[ra] = target_c

            tag = f"{conn}c_{'m' if multi else 's'}"
            if ok_size and size_map:
                smap = dict(size_map)
                def _make_size_fn(sm=smap, cn=conn, ml=multi):
                    def _fn(g: np.ndarray) -> np.ndarray | None:
                        objs = find_objects(g, conn=cn, multicolor=ml, bg=0)
                        if not objs or any(o["size"] not in sm for o in objs):
                            return None
                        res = np.zeros_like(g)
                        for o in objs:
                            c = sm[o["size"]]
                            for r, col in o["pixels"]:
                                res[r, col] = c
                        return res
                    return _fn
                prims.append((f"recolor_by_size_{tag}", _make_size_fn(), "preserve"))

            if ok_rdesc and rank_desc_map:
                rmap = dict(rank_desc_map)
                def _make_rdesc_fn(rm=rmap, cn=conn, ml=multi):
                    def _fn(g: np.ndarray) -> np.ndarray | None:
                        objs = find_objects(g, conn=cn, multicolor=ml, bg=0)
                        if not objs:
                            return None
                        s_desc = sorted({o["size"] for o in objs}, reverse=True)
                        if any(idx not in rm for idx in range(len(s_desc))):
                            return None
                        s_to_c = {s: rm[idx] for idx, s in enumerate(s_desc)}
                        res = np.zeros_like(g)
                        for o in objs:
                            c = s_to_c[o["size"]]
                            for r, col in o["pixels"]:
                                res[r, col] = c
                        return res
                    return _fn
                prims.append((f"recolor_by_rank_desc_{tag}", _make_rdesc_fn(), "preserve"))

            if ok_rasc and rank_asc_map:
                rmap = dict(rank_asc_map)
                def _make_rasc_fn(rm=rmap, cn=conn, ml=multi):
                    def _fn(g: np.ndarray) -> np.ndarray | None:
                        objs = find_objects(g, conn=cn, multicolor=ml, bg=0)
                        if not objs:
                            return None
                        s_asc = sorted({o["size"] for o in objs})
                        if any(idx not in rm for idx in range(len(s_asc))):
                            return None
                        s_to_c = {s: rm[idx] for idx, s in enumerate(s_asc)}
                        res = np.zeros_like(g)
                        for o in objs:
                            c = s_to_c[o["size"]]
                            for r, col in o["pixels"]:
                                res[r, col] = c
                        return res
                    return _fn
                prims.append((f"recolor_by_rank_asc_{tag}", _make_rasc_fn(), "preserve"))
    return prims


def build_parametric_primitives(
    train_in: list[np.ndarray], train_out: list[np.ndarray]
) -> list[tuple[str, Callable, str]]:
    """Instantiate parametric primitives with learned and enumerated parameters."""
    prims: list[tuple[str, Callable, str]] = []

    in_colors = {int(x) for g in train_in for x in np.unique(g)}
    out_colors = {int(x) for g in train_out for x in np.unique(g)}
    novel_all = [
        c for c in range(10)
        if all(np.any(o == c) for o in train_out) and all(not np.any(i == c) for i in train_in)
    ]
    novel_any = [
        c for c in range(10)
        if c not in novel_all and any(np.any(o == c) and not np.any(i == c) for i, o in zip(train_in, train_out))
    ]
    out_freq = Counter(int(x) for o in train_out for x in o.ravel() if x != 0)
    ranked_out_colors = list(dict.fromkeys(
        novel_all + novel_any + [c for c, _ in out_freq.most_common()] + sorted(out_colors - {0}) + [1]
    ))
    all_colors_ordered = list(dict.fromkeys(
        novel_all + novel_any + sorted(in_colors | out_colors) + list(range(10))
    ))

    # Learned scale/tile ratios
    up_ratios = []
    down_ratios = []
    if all(
        o.shape[0] % i.shape[0] == 0 and o.shape[1] % i.shape[1] == 0
        for i, o in zip(train_in, train_out)
    ):
        rs = {(o.shape[0] // i.shape[0], o.shape[1] // i.shape[1]) for i, o in zip(train_in, train_out)}
        if len(rs) == 1:
            ry, rx = next(iter(rs))
            if (ry, rx) != (1, 1) and 1 <= ry <= 10 and 1 <= rx <= 10:
                up_ratios.append((ry, rx))

    if all(
        i.shape[0] % o.shape[0] == 0 and i.shape[1] % o.shape[1] == 0
        for i, o in zip(train_in, train_out)
    ):
        rs = {(i.shape[0] // o.shape[0], i.shape[1] // o.shape[1]) for i, o in zip(train_in, train_out)}
        if len(rs) == 1:
            ry, rx = next(iter(rs))
            if (ry, rx) != (1, 1) and 1 <= ry <= 10 and 1 <= rx <= 10:
                down_ratios.append((ry, rx))

    for ry, rx in list(dict.fromkeys(up_ratios + [(2, 2), (3, 3), (4, 4)])):
        prims.append((f"upscale_{ry}x{rx}", lambda g, ky=ry, kx=rx: p_upscale(g, ky, kx), "change"))
        prims.append((f"tile_{ry}x{rx}", lambda g, ny=ry, nx=rx: p_tile(g, ny, nx), "change"))
    for ny, nx in ((1, 2), (2, 1), (1, 3), (3, 1)):
        if (ny, nx) not in up_ratios:
            prims.append((f"tile_{ny}x{nx}", lambda g, y=ny, x=nx: p_tile(g, y, x), "change"))

    for ry, rx in list(dict.fromkeys(down_ratios + [(2, 2), (3, 3), (4, 4)])):
        prims.append((f"downscale_tl_{ry}x{rx}", lambda g, ky=ry, kx=rx: p_downscale_topleft(g, ky, kx), "change"))
        prims.append((f"downscale_maj_{ry}x{rx}", lambda g, ky=ry, kx=rx: p_downscale_majority(g, ky, kx, False), "change"))
        prims.append((f"downscale_nbmaj_{ry}x{rx}", lambda g, ky=ry, kx=rx: p_downscale_majority(g, ky, kx, True), "change"))

    # Fixed target output dimensions for downscaling when all outputs share a constant shape
    out_shapes = {o.shape for o in train_out}
    if len(out_shapes) == 1:
        th, tw = next(iter(out_shapes))
        prims.append((
            f"downscale_to_{th}x{tw}_maj",
            lambda g, oh=th, ow=tw: (
                p_downscale_majority(g, g.shape[0] // oh, g.shape[1] // ow, True)
                if g.shape[0] % oh == 0 and g.shape[1] % ow == 0 and (g.shape != (oh, ow))
                else None
            ),
            "change",
        ))

    # Learned object size/rank recolourings
    prims.extend(_build_learned_recolor_prims(train_in, train_out))

    # Writer colour primitives (prioritize learned fill colours and output colours)
    for c in ranked_out_colors:
        prims.append((f"fill_enclosed_c{c}", lambda g, col=c: p_fill_enclosed(g, col, 4), "preserve"))
        prims.append((f"fill_enclosed8_c{c}", lambda g, col=c: p_fill_enclosed(g, col, 8), "preserve"))
        prims.append((f"flood_borders_c{c}", lambda g, col=c: p_flood_borders(g, col, 4), "preserve"))
        prims.append((f"outline_hollow_c{c}", lambda g, col=c: p_outline_objects_internal(g, col, True), "preserve"))
        prims.append((f"outline_border_c{c}", lambda g, col=c: p_outline_objects_internal(g, col, False), "preserve"))
        prims.append((f"outline_ext4_c{c}", lambda g, col=c: p_outline_objects_external(g, col, 4), "preserve"))
        prims.append((f"outline_ext8_c{c}", lambda g, col=c: p_outline_objects_external(g, col, 8), "preserve"))
        prims.append((f"fill_rect_c{c}", lambda g, col=c: p_fill_rectangles_interior(g, col), "preserve"))
        for op in ("and", "or", "xor", "uniq1", "diff", "rdiff", "nor", "maj"):
            prims.append((f"cells_{op}_c{c}", lambda g, o=op, col=c: p_cells_op(g, o, col, True), "change"))
        prims.append((f"count_obj4_1xN_c{c}", lambda g, col=c: p_count_objects_bar(g, col, True, 4, False), "change"))
        prims.append((f"count_obj4_Nx1_c{c}", lambda g, col=c: p_count_objects_bar(g, col, False, 4, False), "change"))
        prims.append((f"count_obj8m_1xN_c{c}", lambda g, col=c: p_count_objects_bar(g, col, True, 8, True), "change"))
        prims.append((f"count_obj8m_Nx1_c{c}", lambda g, col=c: p_count_objects_bar(g, col, False, 8, True), "change"))

    # Selector colour primitives over 0..9
    for c in all_colors_ordered:
        prims.append((f"crop_color_c{c}", lambda g, col=c: p_crop_color_bbox(g, col), "change"))
        prims.append((f"crop_inside_color_c{c}", lambda g, col=c: p_crop_inside_color_bbox(g, col), "change"))
        prims.append((f"keep_color_c{c}", lambda g, col=c: p_keep_color(g, col), "preserve"))
        prims.append((f"remove_color_c{c}", lambda g, col=c: p_remove_color(g, col), "preserve"))
        prims.append((f"lines_h_c{c}", lambda g, col=c: p_draw_lines_through_color(g, col, "h"), "preserve"))
        prims.append((f"lines_v_c{c}", lambda g, col=c: p_draw_lines_through_color(g, col, "v"), "preserve"))
        prims.append((f"lines_both_c{c}", lambda g, col=c: p_draw_lines_through_color(g, col, "both"), "preserve"))
        for mode in ("lr", "ud", "both", "rot180", "rot90"):
            prims.append((f"sym_hole_{mode}_c{c}", lambda g, col=c, m=mode: p_sym_fill_hole(g, col, m), "preserve"))

    return prims


# =============================================================================
# Search Engine (solve)
# =============================================================================

def _apply_chain(chain: list[Callable], grids: list[np.ndarray]) -> list[np.ndarray] | None:
    cur = grids
    for fn in chain:
        nxt = []
        for g in cur:
            try:
                r = fn(g)
            except Exception:
                return None
            if not is_valid_grid(r):
                return None
            nxt.append(r)
        cur = nxt
    return cur


def _try_color_map(
    prog_name: str,
    cur_train: list[np.ndarray],
    train_out: list[np.ndarray],
    chain: list[Callable],
    test_in: list[np.ndarray],
) -> tuple[str, list[list[list[int]]]] | None:
    mat = np.zeros((10, 10), dtype=bool)
    for r, o in zip(cur_train, train_out):
        mat[r, o] = True
    row_sums = mat.sum(axis=1)
    if not np.all(row_sums <= 1):
        return None
    cur_test = _apply_chain(chain, test_in)
    if cur_test is None:
        return None
    test_colors = np.zeros(10, dtype=bool)
    for t in cur_test:
        test_colors[t] = True
    covered = row_sums == 1
    if not np.all(covered[test_colors]):
        return None
    lut = np.argmax(mat, axis=1).astype(np.int8)
    mapped_test = [to_list_grid(lut[t]) for t in cur_test]
    pairs = [f"{c}->{int(lut[c])}" for c in range(10) if covered[c] and c != int(lut[c])]
    return (f"{prog_name}+cmap({','.join(pairs)})", mapped_test)


def solve(train_pairs, test_inputs, time_limit=10.0):
    """train_pairs: list of (input_grid, output_grid); test_inputs: list of grids.
    Returns a list of (program_name: str, test_outputs: list[grid]) for every program found within time_limit
    (wall seconds, checked often) that reproduces EVERY train output exactly. Simplest programs first.
    Never raises: any exception inside a candidate program means that program is skipped."""
    try:
        t_start = time.perf_counter()
        deadline = t_start + float(time_limit)
        if not train_pairs or not test_inputs:
            return []

        train_in: list[np.ndarray] = []
        train_out: list[np.ndarray] = []
        for pair in train_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return []
            gi, go = to_grid_array(pair[0]), to_grid_array(pair[1])
            if gi is None or go is None:
                return []
            train_in.append(gi)
            train_out.append(go)

        test_in: list[np.ndarray] = []
        for tg in test_inputs:
            gt = to_grid_array(tg)
            if gt is None:
                return []
            test_in.append(gt)

        all_prims = PRIMITIVES + build_parametric_primitives(train_in, train_out)
        prims_preserve = [p for p in all_prims if p[2] == "preserve"]
        prims_swap = [p for p in all_prims if p[2] == "swap"]
        prims_change = [p for p in all_prims if p[2] == "change"]
        any_shape_changer = bool(prims_swap or prims_change)
        all_out_square = all(o.shape[0] == o.shape[1] for o in train_out)

        solutions: list[tuple[str, list[list[list[int]]]]] = []
        visited: set[tuple[tuple[tuple[int, int], bytes], ...]] = set()

        # Frontier node: (name, chain, train_grids, same_shape, swapped_shape)
        frontier: list[tuple[str, list[Callable], list[np.ndarray], bool, bool]] = [
            ("", [], train_in, all(i.shape == o.shape for i, o in zip(train_in, train_out)), False)
        ]

        for depth in (1, 2, 3):
            if time.perf_counter() >= deadline or not frontier:
                break
            next_frontier: list[tuple[str, list[Callable], list[np.ndarray], bool, bool]] = []

            for p_name_prefix, p_chain, p_train, p_same, p_swap in frontier:
                if time.perf_counter() >= deadline:
                    break
                if depth > 1 and not p_same and not any_shape_changer:
                    continue

                if depth == 3:
                    if p_same:
                        cand_prims = (
                            all_prims if all_out_square else (prims_preserve + prims_change)
                        )
                    elif p_swap:
                        cand_prims = prims_swap + prims_change
                    else:
                        cand_prims = prims_change
                else:
                    cand_prims = all_prims

                for prim_name, prim_fn, _ in cand_prims:
                    if time.perf_counter() >= deadline:
                        break
                    cur_train: list[np.ndarray] = []
                    valid = True
                    same_shape = True
                    swapped_shape = True

                    for i, g in enumerate(p_train):
                        try:
                            r = prim_fn(g)
                        except Exception:
                            valid = False
                            break
                        if not is_valid_grid(r):
                            valid = False
                            break
                        th, tw = train_out[i].shape
                        if r.shape != (th, tw):
                            same_shape = False
                            if depth == 3 or not any_shape_changer:
                                valid = False
                                break
                        if r.shape != (tw, th):
                            swapped_shape = False
                        cur_train.append(r)

                    if not valid:
                        continue

                    state_key = tuple((a.shape, a.tobytes()) for a in cur_train)
                    if state_key in visited:
                        continue
                    visited.add(state_key)

                    full_name = prim_name if not p_name_prefix else f"{p_name_prefix} | {prim_name}"
                    full_chain = p_chain + [prim_fn]

                    if same_shape:
                        exact = all(np.array_equal(r, o) for r, o in zip(cur_train, train_out))
                        if exact:
                            cur_test = _apply_chain(full_chain, test_in)
                            if cur_test is not None:
                                solutions.append((full_name, [to_list_grid(t) for t in cur_test]))
                        else:
                            cm_res = _try_color_map(full_name, cur_train, train_out, full_chain, test_in)
                            if cm_res is not None:
                                solutions.append(cm_res)

                    if depth < 3 and prim_name != "identity":
                        next_frontier.append((full_name, full_chain, cur_train, same_shape, swapped_shape))

            # Prioritize states already matching target shapes when expanding to the next depth
            next_frontier.sort(key=lambda item: (0 if item[3] else (1 if item[4] else 2)))
            frontier = next_frontier

        return solutions
    except Exception:
        return []
    finally:
        _OBJ_CACHE.clear()
        _SPLIT_CACHE.clear()
        _ENC_CACHE.clear()


# =============================================================================
# Registry List of Parameter-Free Primitives (name, fn, shape_mode)
# =============================================================================

PRIMITIVES: list[tuple[str, Callable[[np.ndarray], np.ndarray | None], str]] = [
    # 1. Geometry
    ("identity", p_identity, "preserve"),
    ("rot90", p_rot90, "swap"),
    ("rot180", p_rot180, "preserve"),
    ("rot270", p_rot270, "swap"),
    ("flip_lr", p_flip_lr, "preserve"),
    ("flip_ud", p_flip_ud, "preserve"),
    ("transpose", p_transpose, "swap"),
    ("anti_transpose", p_anti_transpose, "swap"),
    ("crop_border1", p_crop_border1, "change"),
    ("left_half", p_left_half, "change"),
    ("right_half", p_right_half, "change"),
    ("top_half", p_top_half, "change"),
    ("bottom_half", p_bottom_half, "change"),
    # 2. Cropping
    ("crop_non_bg", lambda g: p_crop_non_bg(g, 0), "change"),
    ("crop_non_bg_auto", lambda g: p_crop_non_bg(g, None), "change"),
    ("crop_largest_4s", lambda g: _crop_obj_by_size(g, True, 4, False, False, 0), "change"),
    ("crop_smallest_4s", lambda g: _crop_obj_by_size(g, False, 4, False, False, 0), "change"),
    ("crop_largest_8s", lambda g: _crop_obj_by_size(g, True, 8, False, False, 0), "change"),
    ("crop_smallest_8s", lambda g: _crop_obj_by_size(g, False, 8, False, False, 0), "change"),
    ("crop_largest_4m", lambda g: _crop_obj_by_size(g, True, 4, True, False, 0), "change"),
    ("crop_smallest_4m", lambda g: _crop_obj_by_size(g, False, 4, True, False, 0), "change"),
    ("crop_largest_8m", lambda g: _crop_obj_by_size(g, True, 8, True, False, 0), "change"),
    ("crop_smallest_8m", lambda g: _crop_obj_by_size(g, False, 8, True, False, 0), "change"),
    ("crop_largest_4m_masked", lambda g: _crop_obj_by_size(g, True, 4, True, True, 0), "change"),
    ("crop_largest_8m_masked", lambda g: _crop_obj_by_size(g, True, 8, True, True, 0), "change"),
    ("crop_second_largest", p_crop_second_largest, "change"),
    ("crop_unique_color_4", lambda g: p_crop_unique_color_obj(g, 4, False), "change"),
    ("crop_unique_color_8", lambda g: p_crop_unique_color_obj(g, 8, False), "change"),
    ("crop_unique_shape_4s", lambda g: p_crop_unique_shape_obj(g, 4, False, False), "change"),
    ("crop_unique_shape_8m", lambda g: p_crop_unique_shape_obj(g, 8, True, False), "change"),
    ("crop_unique_shape_rot", lambda g: p_crop_unique_shape_obj(g, 4, False, True), "change"),
    # 3. Scaling and Tiling
    ("compress_unique_lines", p_compress_unique_lines, "change"),
    ("fractal_upscale", lambda g: p_fractal_upscale(g, False), "change"),
    ("fractal_upscale_recolor", lambda g: p_fractal_upscale(g, True), "change"),
    ("mirror_tile_2x2", lambda g: p_mirror_tile_2x2(g, 0), "change"),
    ("mirror_tile_2x2_inv", lambda g: p_mirror_tile_2x2(g, 1), "change"),
    ("mirror_tile_2x2_lr", lambda g: p_mirror_tile_2x2(g, 2), "change"),
    ("mirror_tile_2x2_ud", lambda g: p_mirror_tile_2x2(g, 3), "change"),
    ("hconcat_fliplr", lambda g: p_hconcat_fliplr(g, False), "change"),
    ("hconcat_fliplr_rev", lambda g: p_hconcat_fliplr(g, True), "change"),
    ("vconcat_flipud", lambda g: p_vconcat_flipud(g, False), "change"),
    ("vconcat_flipud_rev", lambda g: p_vconcat_flipud(g, True), "change"),
    # 4. Separator Lines
    ("cells_overlay", lambda g: p_cells_overlay(g, False), "change"),
    ("cells_overlay_rev", lambda g: p_cells_overlay(g, True), "change"),
    ("pick_unique_cell", p_pick_unique_cell, "change"),
    ("pick_most_nonbg_cell", lambda g: p_pick_extreme_nonbg_cell(g, True), "change"),
    ("pick_fewest_nonbg_cell", lambda g: p_pick_extreme_nonbg_cell(g, False), "change"),
    ("condense_cells", p_condense_cells, "change"),
    ("remove_separators", p_remove_separators, "change"),
    # 5. Physics and Fill
    ("gravity_down", lambda g: p_gravity(g, "down"), "preserve"),
    ("gravity_up", lambda g: p_gravity(g, "up"), "preserve"),
    ("gravity_left", lambda g: p_gravity(g, "left"), "preserve"),
    ("gravity_right", lambda g: p_gravity(g, "right"), "preserve"),
    ("fill_enclosed_by_neighbor", p_fill_enclosed_by_neighbor, "preserve"),
    ("connect_aligned_both", lambda g: p_connect_aligned_same_color(g, "both"), "preserve"),
    ("connect_aligned_h", lambda g: p_connect_aligned_same_color(g, "h"), "preserve"),
    ("connect_aligned_v", lambda g: p_connect_aligned_same_color(g, "v"), "preserve"),
    ("extend_ray_down", lambda g: p_extend_pixels(g, "down", False, False), "preserve"),
    ("extend_ray_up", lambda g: p_extend_pixels(g, "up", False, False), "preserve"),
    ("extend_ray_right", lambda g: p_extend_pixels(g, "right", False, False), "preserve"),
    ("extend_ray_left", lambda g: p_extend_pixels(g, "left", False, False), "preserve"),
    ("extend_dots_all4", lambda g: p_extend_pixels(g, "all4", True, False), "preserve"),
    ("bridge_obstacles_all4", lambda g: p_extend_pixels(g, "all4", False, True), "preserve"),
    # 6. Symmetry Completion
    ("sym_complete_lr", lambda g: p_sym_complete(g, "lr", False), "preserve"),
    ("sym_complete_ud", lambda g: p_sym_complete(g, "ud", False), "preserve"),
    ("sym_complete_both", lambda g: p_sym_complete(g, "both", False), "preserve"),
    ("sym_complete_rot180", lambda g: p_sym_complete(g, "rot180", False), "preserve"),
    ("sym_complete_rot90", lambda g: p_sym_complete(g, "rot90", False), "preserve"),
    ("sym_complete_transpose", lambda g: p_sym_complete(g, "transpose", False), "preserve"),
    ("sym_complete_antitranspose", lambda g: p_sym_complete(g, "antitranspose", False), "preserve"),
    ("sym_bbox_lr", lambda g: p_sym_complete(g, "lr", True), "preserve"),
    ("sym_bbox_ud", lambda g: p_sym_complete(g, "ud", True), "preserve"),
    ("sym_bbox_both", lambda g: p_sym_complete(g, "both", True), "preserve"),
    # 7. Colour Ops
    ("swap_top2_colors", lambda g: p_swap_colors_by_freq(g, True), "preserve"),
    ("swap_most_least_colors", lambda g: p_swap_colors_by_freq(g, False), "preserve"),
    ("recolor_rank_desc_4s", lambda g: p_recolor_objects_canonical(g, "rank_desc", 4, False), "preserve"),
    ("recolor_rank_asc_4s", lambda g: p_recolor_objects_canonical(g, "rank_asc", 4, False), "preserve"),
    ("recolor_rank_desc_8m", lambda g: p_recolor_objects_canonical(g, "rank_desc", 8, True), "preserve"),
    ("recolor_size_clamp_4s", lambda g: p_recolor_objects_canonical(g, "size_clamp", 4, False), "preserve"),
    # 8. Denoise
    ("remove_isolated_4", lambda g: p_remove_isolated_pixels(g, 4, True), "preserve"),
    ("remove_isolated_8", lambda g: p_remove_isolated_pixels(g, 8, False), "preserve"),
    ("remove_small_lt2_4s", lambda g: p_remove_small_objects(g, 2, 4, False), "preserve"),
    ("remove_small_lt3_4s", lambda g: p_remove_small_objects(g, 3, 4, False), "preserve"),
    ("remove_small_lt4_4s", lambda g: p_remove_small_objects(g, 4, 4, False), "preserve"),
    ("remove_small_lt3_8m", lambda g: p_remove_small_objects(g, 3, 8, True), "preserve"),
    ("keep_largest_in_place_4s", lambda g: p_keep_extreme_object_in_place(g, True, 4, False), "preserve"),
    ("keep_largest_in_place_8m", lambda g: p_keep_extreme_object_in_place(g, True, 8, True), "preserve"),
    ("keep_smallest_in_place_4s", lambda g: p_keep_extreme_object_in_place(g, False, 4, False), "preserve"),
    # 9. Counting Outputs
    ("most_freq_color_1x1", lambda g: p_extreme_freq_color_1x1(g, True), "change"),
    ("least_freq_color_1x1", lambda g: p_extreme_freq_color_1x1(g, False), "change"),
    # 10. Outline and Rectangle Fill
    ("outline_hollow_orig", lambda g: p_outline_objects_internal(g, -1, True), "preserve"),
    ("fill_rect_interior_self", lambda g: p_fill_rectangles_interior(g, None), "preserve"),
    ("fill_object_bboxes_4s", lambda g: p_fill_object_bboxes(g, 4, False), "preserve"),
    ("fill_object_bboxes_8m", lambda g: p_fill_object_bboxes(g, 8, True), "preserve"),
]


if __name__ == "__main__":
    # Self-test on made-up toy grids
    toy_train = [
        ([[0, 0, 0], [0, 2, 3], [0, 0, 0]], [[3, 2], [0, 0]]),
        ([[0, 2, 0, 0], [0, 3, 2, 0]], [[0, 2], [2, 3]]),
    ]
    toy_test = [[[0, 0, 0], [2, 3, 0], [0, 2, 0]]]
    res = solve(toy_train, toy_test, time_limit=2.0)
    assert any(out == [[[0, 3], [2, 0]]] for _, out in res), f"Unexpected result: {res}"

    # Enclosed fill + colour map toy test
    enc_train = [
        ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 1, 1], [1, 4, 1], [1, 1, 1]]),
        ([[2, 2, 2, 2], [2, 0, 0, 2], [2, 2, 2, 2]], [[2, 2, 2, 2], [2, 4, 4, 2], [2, 2, 2, 2]]),
    ]
    enc_test = [[[3, 3, 3], [3, 0, 3], [3, 3, 3]]]
    res_enc = solve(enc_train, enc_test, time_limit=2.0)
    assert any(out == [[[3, 3, 3], [3, 4, 3], [3, 3, 3]]] for _, out in res_enc)
    print("Self-test passed:", len(res), len(res_enc))
