"""Recipe: abs_cumsum_rows (cumulative sum of |x| over time, src/abscumsum.almd)."""
import numpy as np

NAME = "abs_cumsum_rows"
MODULE = "abscumsum"
CALL = "abscumsum.abs_cumsum_rows(x)"
TOL = 1e-9
SEED = 20260900


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.cumsum(np.abs(x), axis=0)
