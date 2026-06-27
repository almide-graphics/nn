"""Recipe: canberra_dist_rows (row-wise Canberra distance, src/canberra.almd)."""
import numpy as np

NAME = "canberra_dist_rows"
MODULE = "canberra"
CALL = "canberra.canberra_dist_rows(a, b)"
TOL = 1e-9
SEED = 20260980


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    num = np.abs(a - b)
    den = np.abs(a) + np.abs(b) + 1e-12
    return (num / den).sum(axis=1, keepdims=True)
