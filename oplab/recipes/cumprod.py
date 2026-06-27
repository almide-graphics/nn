"""Recipe: cumprod_rows (cumulative product over time, src/cumprod.almd)."""
import numpy as np

NAME = "cumprod_rows"
MODULE = "cumprod"
CALL = "cumprod.cumprod_rows(x)"
TOL = 1e-9
SEED = 20260901


def make_inputs(rng):
    # scaled down so the running product stays in a tame range
    return {"x": ("matrix", 0.9 * rng.standard_normal((5, 4)))}


def reference(x):
    return np.cumprod(x, axis=0)
