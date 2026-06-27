"""Recipe: weighted_mean_rows (per-row weighted average with explicit weights, src/weightedmean.almd)."""
import numpy as np

NAME = "weighted_mean_rows"
MODULE = "weightedmean"
CALL = "weightedmean.weighted_mean_rows(x, w)"
TOL = 1e-9
SEED = 20261043


def make_inputs(rng):
    x = rng.standard_normal((5, 4))
    w = rng.standard_normal((5, 4)) ** 2  # non-negative weights
    return {"x": ("matrix", x), "w": ("matrix", w)}


def reference(x, w):
    num = (x * w).sum(axis=1, keepdims=True)
    den = w.sum(axis=1, keepdims=True) + 1e-12
    return num / den
