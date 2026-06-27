"""Recipe: cummax_minus_cummin_rows (running peak-to-trough range over time, src/cummaxminuscummin.almd)."""
import numpy as np

NAME = "cummax_minus_cummin_rows"
MODULE = "cummaxminuscummin"
CALL = "cummaxminuscummin.cummax_minus_cummin_rows(x)"
TOL = 1e-9
SEED = 20260945


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.maximum.accumulate(x, axis=0) - np.minimum.accumulate(x, axis=0)
