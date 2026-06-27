"""Recipe: cumsum_rows (prefix sum over time, src/cumsum.almd)."""
import numpy as np

NAME = "cumsum"
MODULE = "cumsum"
CALL = "cumsum.cumsum_rows(x)"
TOL = 1e-9
SEED = 20260701


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((6, 4)))}


def reference(x):
    return np.cumsum(x, axis=0)
