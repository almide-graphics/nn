"""Recipe: cumsum_sign_change_rows (running zero-crossing count over time, src/cumsignchange.almd)."""
import numpy as np

NAME = "cumsum_sign_change_rows"
MODULE = "cumsignchange"
CALL = "cumsignchange.cumsum_sign_change_rows(x)"
TOL = 1e-9
SEED = 20260953


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    s = np.sign(x)
    sc = np.zeros_like(x, dtype=float)
    sc[1:] = (s[1:] != s[:-1]).astype(float)
    return np.cumsum(sc, axis=0)
