"""Recipe: softmax_weighted_var_rows (variance of values under softmax attention weights, src/softmaxwvar.almd)."""
import numpy as np

NAME = "softmax_weighted_var_rows"
MODULE = "softmaxwvar"
CALL = "softmaxwvar.softmax_weighted_var_rows(x, w)"
TOL = 1e-9
SEED = 20260985


def make_inputs(rng):
    return {
        "x": ("matrix", rng.standard_normal((5, 4))),
        "w": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(x, w):
    e = np.exp(w - w.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    mu = (p * x).sum(axis=1, keepdims=True)
    return (p * (x - mu) * (x - mu)).sum(axis=1, keepdims=True)
