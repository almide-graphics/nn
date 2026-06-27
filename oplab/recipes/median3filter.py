"""Recipe: median3_filter_rows (length-3 median filter over time, despiking, src/median3filter.almd)."""
import numpy as np

NAME = "median3_filter_rows"
MODULE = "median3filter"
CALL = "median3filter.median3_filter_rows(x)"
TOL = 1e-9
SEED = 20261042


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    out = np.empty((t, d), dtype=float)
    for i in range(t):
        lo = max(i - 1, 0)
        hi = min(i + 1, t - 1)
        a, b, c = x[lo], x[i], x[hi]
        hi3 = np.maximum(np.maximum(a, b), c)
        lo3 = np.minimum(np.minimum(a, b), c)
        out[i] = a + b + c - hi3 - lo3
    return out
