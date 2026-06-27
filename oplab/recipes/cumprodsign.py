"""Recipe: cumprod_sign_rows (running product of signs over time, src/cumprodsign.almd)."""
import numpy as np

NAME = "cumprod_sign_rows"
MODULE = "cumprodsign"
CALL = "cumprodsign.cumprod_sign_rows(x)"
TOL = 1e-9
SEED = 20260947


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.cumprod(np.sign(x), axis=0)
