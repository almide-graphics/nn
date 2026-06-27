"""Recipe: cov_rows (row-wise population covariance, src/covrows.almd)."""
import numpy as np

NAME = "cov_rows"
MODULE = "covrows"
CALL = "covrows.cov_rows(a, b)"
TOL = 1e-9
SEED = 20260886


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    ma = a.mean(axis=1, keepdims=True)
    mb = b.mean(axis=1, keepdims=True)
    return ((a - ma) * (b - mb)).mean(axis=1, keepdims=True)
