"""Recipe: lag1_rows (one-step causal delay over time, src/lag1.almd)."""
import numpy as np

NAME = "lag1_rows"
MODULE = "lag1"
CALL = "lag1.lag1_rows(x)"
TOL = 1e-9
SEED = 20260903


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.vstack([np.zeros((1, x.shape[1])), x[:-1]])
