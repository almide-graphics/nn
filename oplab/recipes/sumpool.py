"""Recipe: sumpool_rows (row-wise sum, src/sumpool.almd)."""
import numpy as np

NAME = "sumpool_rows"
MODULE = "sumpool"
CALL = "sumpool.sumpool_rows(x)"
TOL = 1e-9
SEED = 20260731


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x.sum(axis=1, keepdims=True)
