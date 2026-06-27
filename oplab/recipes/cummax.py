"""Recipe: cummax_rows (cumulative max over time / down rows, src/cummax.almd)."""
import numpy as np

NAME = "cummax_rows"
MODULE = "cummax"
CALL = "cummax.cummax_rows(x)"
TOL = 1e-9
SEED = 20260897


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.maximum.accumulate(x, axis=0)
