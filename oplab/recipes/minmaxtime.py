"""Recipe: minmax_time_rows (min-max normalization over time, src/minmaxtime.almd)."""
import numpy as np

NAME = "minmax_time_rows"
MODULE = "minmaxtime"
CALL = "minmaxtime.minmax_time_rows(x)"
TOL = 1e-9
SEED = 20260912


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    mn = x.min(axis=0, keepdims=True)
    mx = x.max(axis=0, keepdims=True)
    return (x - mn) / (mx - mn)
