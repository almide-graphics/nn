"""Recipe: abs_max_rows (row-wise L-infinity norm, src/absmax.almd)."""
import numpy as np

NAME = "abs_max_rows"
MODULE = "absmax"
CALL = "absmax.abs_max_rows(x)"
TOL = 1e-9
SEED = 20260728


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.abs(x).max(axis=1, keepdims=True)
