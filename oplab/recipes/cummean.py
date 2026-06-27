"""Recipe: cummean_rows (cumulative mean over time, src/cummean.almd)."""
import numpy as np

NAME = "cummean_rows"
MODULE = "cummean"
CALL = "cummean.cummean_rows(x)"
TOL = 1e-9
SEED = 20260902


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t = x.shape[0]
    return np.cumsum(x, axis=0) / np.arange(1, t + 1).reshape(-1, 1)
