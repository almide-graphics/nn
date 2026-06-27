"""Recipe: cumsum_sq_rows (cumulative sum of squares over time, src/cumsumsq.almd)."""
import numpy as np

NAME = "cumsum_sq_rows"
MODULE = "cumsumsq"
CALL = "cumsumsq.cumsum_sq_rows(x)"
TOL = 1e-9
SEED = 20260905


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.cumsum(x ** 2, axis=0)
