"""Recipe: mae_rows (row-wise mean absolute error, src/maerows.almd)."""
import numpy as np

NAME = "mae_rows"
MODULE = "maerows"
CALL = "maerows.mae_rows(a, b)"
TOL = 1e-9
SEED = 20260874


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return np.abs(a - b).mean(axis=1, keepdims=True)
