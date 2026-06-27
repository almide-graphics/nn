"""Recipe: range_rows (row-wise peak-to-peak, src/rangerows.almd)."""
import numpy as np

NAME = "range_rows"
MODULE = "rangerows"
CALL = "rangerows.range_rows(x)"
TOL = 1e-9
SEED = 20260895


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True)
