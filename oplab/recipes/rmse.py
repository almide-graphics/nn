"""Recipe: rmse_rows (row-wise root mean squared error, src/rmse.almd)."""
import numpy as np

NAME = "rmse_rows"
MODULE = "rmse"
CALL = "rmse.rmse_rows(a, b)"
TOL = 1e-9
SEED = 20260892


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return np.sqrt(((a - b) ** 2).mean(axis=1, keepdims=True))
