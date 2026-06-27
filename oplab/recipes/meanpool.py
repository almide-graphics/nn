"""Recipe: global_meanpool_rows (row-wise mean, src/meanpool.almd)."""
import numpy as np

NAME = "global_meanpool_rows"
MODULE = "meanpool"
CALL = "meanpool.global_meanpool_rows(x)"
TOL = 1e-9
SEED = 20260724


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x.mean(axis=1, keepdims=True)
