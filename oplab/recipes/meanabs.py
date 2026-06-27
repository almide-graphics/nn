"""Recipe: mean_abs_rows (row-wise mean absolute value, src/meanabs.almd)."""
import numpy as np

NAME = "mean_abs_rows"
MODULE = "meanabs"
CALL = "meanabs.mean_abs_rows(x)"
TOL = 1e-9
SEED = 20260896


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.abs(x).mean(axis=1, keepdims=True)
