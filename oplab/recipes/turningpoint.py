"""Recipe: turning_point_count_rows (per-row count of local extrema / direction reversals, src/turningpoint.almd)."""
import numpy as np

NAME = "turning_point_count_rows"
MODULE = "turningpoint"
CALL = "turningpoint.turning_point_count_rows(x)"
TOL = 1e-9
SEED = 20261002


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    left = x[:, 1:-1] - x[:, :-2]
    right = x[:, 2:] - x[:, 1:-1]
    tp = ((left > 0) & (right < 0)) | ((left < 0) & (right > 0))
    return tp.sum(axis=1, keepdims=True).astype(float)
