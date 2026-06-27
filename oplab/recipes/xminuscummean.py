"""Recipe: x_minus_cummean_rows (deviation from the running mean, src/xminuscummean.almd)."""
import numpy as np

NAME = "x_minus_cummean_rows"
MODULE = "xminuscummean"
CALL = "xminuscummean.x_minus_cummean_rows(x)"
TOL = 1e-9
SEED = 20260944


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    n = np.arange(1, x.shape[0] + 1).reshape(-1, 1)
    cummean = np.cumsum(x, axis=0) / n
    return x - cummean
