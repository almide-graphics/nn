"""Recipe: mse_rows (row-wise mean squared error, src/mserows.almd)."""
import numpy as np

NAME = "mse_rows"
MODULE = "mserows"
CALL = "mserows.mse_rows(a, b)"
TOL = 1e-9
SEED = 20260873


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return ((a - b) ** 2).mean(axis=1, keepdims=True)
