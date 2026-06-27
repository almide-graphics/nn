"""Recipe: cumrms_rows (cumulative root-mean-square over time, src/cumrms.almd)."""
import numpy as np

NAME = "cumrms_rows"
MODULE = "cumrms"
CALL = "cumrms.cumrms_rows(x)"
TOL = 1e-9
SEED = 20261008


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    n = np.arange(1, x.shape[0] + 1).reshape(-1, 1)
    return np.sqrt(np.cumsum(x * x, axis=0) / n)
