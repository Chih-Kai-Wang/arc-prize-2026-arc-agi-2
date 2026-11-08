"""Pure-Python D4 transforms and exact subgrid matches for ARC grids.

Generic utilities for a research solver. No task IDs or expected test outputs.
"""
from typing import List, Tuple

Grid = List[List[int]]


def rot90(g: Grid) -> Grid:
    """Rotate a rectangular grid counterclockwise by 90 degrees."""
    return [list(row) for row in zip(*g)][::-1]


def flip_lr(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def d4(g: Grid):
    """Yield (name, transformed_grid), including all 8 transform labels."""
    a = [row[:] for row in g]
    for rot in range(4):
        yield f'R{90*rot}', a
        yield f'flipLR_R{90*rot}', flip_lr(a)
        a = rot90(a)


def occurrences(big: Grid, patch: Grid):
    """All exact matches of any D4 transform of patch in big; may include equivalent matches."""
    H, W = len(big), len(big[0])
    hits=[]
    for label, transformed in d4(patch):
        h, w = len(transformed), len(transformed[0])
        if h > H or w > W:
            continue
        for r in range(H-h+1):
            for c in range(W-w+1):
                if all(big[r+k][c:c+w] == transformed[k] for k in range(h)):
                    hits.append({'orientation':label,'row':r,'col':c,'height':h,'width':w})
    return hits


def validate_grid(g):
    if not isinstance(g,list) or not g or not isinstance(g[0],list):
        raise ValueError('Expected nonempty grid of rows')
    h,w=len(g),len(g[0])
    if not (1<=h<=30 and 1<=w<=30):
        raise ValueError(f'Grid dimensions {h}x{w} outside ARC 1..30')
    if any(len(row)!=w or any(not isinstance(v,int) or isinstance(v,bool) or not(0<=v<=9) for v in row) for row in g):
        raise ValueError('Grid must be rectangular with integer colors 0..9')
    return h,w

