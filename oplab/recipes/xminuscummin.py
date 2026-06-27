"""Recipe: x_minus_cummin_rows (rise above the running minimum over time, src/xminuscummin.almd)."""
import numpy as np

NAME = "x_minus_cummin_rows"
MODULE = "xminuscummin"
CALL = "xminuscummin.x_minus_cummin_rows(x)"
TOL = 1e-9
SEED = 20260926


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x - np.minimum.accumulate(x, axis=0)
