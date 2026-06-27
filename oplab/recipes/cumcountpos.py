"""Recipe: cumcount_pos_rows (running count of positive entries over time, src/cumcountpos.almd)."""
import numpy as np

NAME = "cumcount_pos_rows"
MODULE = "cumcountpos"
CALL = "cumcountpos.cumcount_pos_rows(x)"
TOL = 1e-9
SEED = 20260972


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.cumsum((x > 0).astype(float), axis=0)
