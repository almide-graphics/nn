"""Recipe: softmax_max_rows (maximum softmax probability / MSP confidence per row, src/softmaxmax.almd)."""
import numpy as np

NAME = "softmax_max_rows"
MODULE = "softmaxmax"
CALL = "softmaxmax.softmax_max_rows(x)"
TOL = 1e-9
SEED = 20260969


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return p.max(axis=1, keepdims=True)
