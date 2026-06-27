"""Recipe: cummax_minus_x_rows (drawdown from the running maximum over time, src/cummaxminusx.almd)."""
import numpy as np

NAME = "cummax_minus_x_rows"
MODULE = "cummaxminusx"
CALL = "cummaxminusx.cummax_minus_x_rows(x)"
TOL = 1e-9
SEED = 20260920


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.maximum.accumulate(x, axis=0) - x
