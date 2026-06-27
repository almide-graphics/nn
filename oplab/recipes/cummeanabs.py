"""Recipe: cummean_abs_rows (cumulative mean of |x| over time, src/cummeanabs.almd)."""
import numpy as np

NAME = "cummean_abs_rows"
MODULE = "cummeanabs"
CALL = "cummeanabs.cummean_abs_rows(x)"
TOL = 1e-9
SEED = 20260914


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t = x.shape[0]
    return np.cumsum(np.abs(x), axis=0) / np.arange(1, t + 1).reshape(-1, 1)
