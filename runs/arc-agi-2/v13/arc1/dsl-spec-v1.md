# Spec: clean-room CPU DSL solver for ARC grids (ARC-1 "D", v1)

Write ONE self-contained Python 3.11 module `dsl.py`. Imports: Python stdlib and numpy only. No file/network I/O, no
subprocess, no global state between calls. It must be deterministic.

## Data format
A grid is a list of lists of ints 0..9 (rows), 1x1 up to 30x30, rectangular. A task gives 2-10 train pairs
(input grid, output grid) and 1-3 test input grids. Colour 0 is usually (not always) background.

## API (exact)
```python
def solve(train_pairs, test_inputs, time_limit=10.0):
    """train_pairs: list of (input_grid, output_grid); test_inputs: list of grids.
    Returns a list of (program_name: str, test_outputs: list[grid]) for every program found within time_limit
    (wall seconds, checked often) that reproduces EVERY train output exactly. Simplest programs first.
    Never raises: any exception inside a candidate program means that program is skipped."""
```
Each returned test output must be a valid grid (rectangular, 1..30 per side, ints 0..9). Return [] if nothing verifies.

## Search
- Enumerate compositions of up to 3 unary grid->grid primitives (breadth-first, depth 1, then 2, then 3), pruning
  any composition whose intermediate result on some train input is invalid or identical to an earlier one
  (dedupe on the tuple of intermediate train results).
- After each structural composition T, also try a learned pixelwise colour map m: if one consistent map
  colour->colour makes m(T(input)) == output for every train pair (and covers every colour seen in the test inputs'
  T result, otherwise skip), emit T+m.
- Parametric primitives take parameters learned from the train pairs (e.g. the colour to fill with = the colour that
  appears in all outputs but in no inputs; scale factor = output/input size ratio) or enumerated over small ranges
  (colours 0..9, factors 2..4).
- Cheap early rejection: if a composition's result shape on a train input differs from the train output shape and
  no later primitive can change shape, stop extending it.

## Primitives (implement as many as you can, robustly; you may add others of similar generality)
1. Geometry: identity, rot90/180/270, flip LR/UD, transpose, anti-transpose.
2. Cropping: crop to bounding box of all non-background cells; crop to bbox of the largest / smallest object
   (4- and 8-connected, single-colour and multi-colour variants); crop to the object with a unique colour; crop to
   the object whose shape occurs once; crop to the bbox of colour c.
3. Scaling/tiling: upscale by k (each cell -> kxk); downscale by k (block majority or block top-left);
   tile nxm; mirror-tile 2x2 (grid | flipLR / flipUD | rot180); hconcat/vconcat with a flipped copy.
4. Grids with separator lines (full rows/cols of one colour): split into cells; combine cells with AND / OR / XOR /
   "left minus right" producing colour c (c learned); pick the cell that differs from the others / has most or fewest
   non-background cells.
5. Physics/fill: gravity of non-background cells in 4 directions; fill enclosed background regions (not connected to
   the border) with colour c; flood background from borders; draw horizontal/vertical lines through each pixel of
   colour c to the borders; extend pixels until hitting non-background.
6. Symmetry completion: make the grid symmetric (LR, UD, both, rot90) by copying non-background cells onto background
   cells, and the variant where a single colour marks the hole to be filled.
7. Colour ops: keep only colour c (others -> background); remove colour c; swap the two most frequent non-bg colours;
   recolour every object by its size rank / by size (1 cell, 2 cells, ...) with learned colours.
8. Denoise: remove isolated single pixels; remove objects smaller than k.
9. Counting outputs: 1x1 grid of the most / least frequent non-background colour; 1xN or Nx1 grid of colour c with N =
   number of objects.
10. Outline: draw the outline (border cells) of each object in colour c; fill the interior of each rectangle.

## Quality bar
- Correctness first: a program must reproduce all train pairs exactly; do not special-case anything.
- Keep total code under ~900 lines, readable, each primitive a small function, registry list at the bottom.
- Include a tiny self-test under `if __name__ == "__main__":` using made-up toy grids only.

Return only the complete `dsl.py` source in one ```python block.
