"""Recipe: hist_intersection_rows (histogram intersection kernel sum(min(a,b)), src/histintersect.almd)."""
import numpy as np

NAME = "hist_intersection_rows"
MODULE = "histintersect"
CALL = "histintersect.hist_intersection_rows(a, b)"
TOL = 1e-9
SEED = 20261051


def make_inputs(rng):
    a = rng.standard_normal((5, 4)) ** 2  # non-negative histograms
    b = rng.standard_normal((5, 4)) ** 2
    return {"a": ("matrix", a), "b": ("matrix", b)}


def reference(a, b):
    return np.minimum(a, b).sum(axis=1, keepdims=True)
