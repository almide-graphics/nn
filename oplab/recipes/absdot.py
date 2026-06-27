"""Recipe: abs_dot_rows (row-wise absolute dot product, src/absdot.almd)."""
import numpy as np

NAME = "abs_dot_rows"
MODULE = "absdot"
CALL = "absdot.abs_dot_rows(a, b)"
TOL = 1e-9
SEED = 20260890


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return np.abs((a * b).sum(axis=1, keepdims=True))
