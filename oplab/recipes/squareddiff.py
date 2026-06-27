"""Recipe: squared_diff_rows (squared first difference over time, src/squareddiff.almd)."""
import numpy as np

NAME = "squared_diff_rows"
MODULE = "squareddiff"
CALL = "squareddiff.squared_diff_rows(x)"
TOL = 1e-9
SEED = 20260952


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    out = np.zeros_like(x, dtype=float)
    d = x[1:] - x[:-1]
    out[1:] = d * d
    return out
