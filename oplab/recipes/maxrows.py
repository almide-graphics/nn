"""Recipe: max_rows (row-wise maximum, src/maxrows.almd)."""
import numpy as np

NAME = "max_rows"
MODULE = "maxrows"
CALL = "maxrows.max_rows(x)"
TOL = 1e-9
SEED = 20260732


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x.max(axis=1, keepdims=True)
