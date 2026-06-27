"""Recipe: abs_diff_rows (absolute first difference over time, src/absdiff.almd)."""
import numpy as np

NAME = "abs_diff_rows"
MODULE = "absdiff"
CALL = "absdiff.abs_diff_rows(x)"
TOL = 1e-9
SEED = 20260951


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    out = np.zeros_like(x, dtype=float)
    out[1:] = np.abs(x[1:] - x[:-1])
    return out
