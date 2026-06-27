"""Recipe: minmax_rows (row-internal min-max normalization, axis=1, src/minmaxrows.almd)."""
import numpy as np

NAME = "minmax_rows"
MODULE = "minmaxrows"
CALL = "minmaxrows.minmax_rows(x)"
TOL = 1e-9
SEED = 20260998


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    mn = x.min(axis=1, keepdims=True)
    mx = x.max(axis=1, keepdims=True)
    return (x - mn) / (mx - mn + 1e-12)
