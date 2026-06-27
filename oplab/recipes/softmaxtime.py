"""Recipe: softmax_time_rows (softmax over the time axis, src/softmaxtime.almd)."""
import numpy as np

NAME = "softmax_time_rows"
MODULE = "softmaxtime"
CALL = "softmaxtime.softmax_time_rows(x)"
TOL = 1e-9
SEED = 20260911


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    e = np.exp(x - x.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)
