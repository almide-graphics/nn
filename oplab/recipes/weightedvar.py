"""Recipe: weighted_var_rows (per-row weighted variance with explicit weights, src/weightedvar.almd)."""
import numpy as np

NAME = "weighted_var_rows"
MODULE = "weightedvar"
CALL = "weightedvar.weighted_var_rows(x, w)"
TOL = 1e-9
SEED = 20261044


def make_inputs(rng):
    x = rng.standard_normal((5, 4))
    w = rng.standard_normal((5, 4)) ** 2  # non-negative weights
    return {"x": ("matrix", x), "w": ("matrix", w)}


def reference(x, w):
    sw = w.sum(axis=1, keepdims=True) + 1e-12
    mu = (x * w).sum(axis=1, keepdims=True) / sw
    return (w * (x - mu) ** 2).sum(axis=1, keepdims=True) / sw
