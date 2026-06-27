"""Recipe: squared_l2_dist_rows (row-wise squared Euclidean distance, src/sql2dist.almd)."""
import numpy as np

NAME = "squared_l2_dist_rows"
MODULE = "sql2dist"
CALL = "sql2dist.squared_l2_dist_rows(a, b)"
TOL = 1e-9
SEED = 20260883


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return ((a - b) ** 2).sum(axis=1, keepdims=True)
