"""Recipe: argmin_rows (per-row hard argmin index, src/argmin.almd)."""
import numpy as np

NAME = "argmin_rows"
MODULE = "argmin"
CALL = "argmin.argmin_rows(x)"
TOL = 1e-9
SEED = 20261050


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.argmin(x, axis=1, keepdims=True).astype(float)
