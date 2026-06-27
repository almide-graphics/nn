"""Recipe: l2_dist_rows (row-wise Euclidean distance, src/l2dist.almd)."""
import numpy as np

NAME = "l2_dist_rows"
MODULE = "l2dist"
CALL = "l2dist.l2_dist_rows(a, b)"
TOL = 1e-9
SEED = 20260880


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1, keepdims=True))
