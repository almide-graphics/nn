"""Recipe: l1_dist_rows (row-wise Manhattan distance, src/l1dist.almd)."""
import numpy as np

NAME = "l1_dist_rows"
MODULE = "l1dist"
CALL = "l1dist.l1_dist_rows(a, b)"
TOL = 1e-9
SEED = 20260881


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return np.abs(a - b).sum(axis=1, keepdims=True)
