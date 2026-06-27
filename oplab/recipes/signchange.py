"""Recipe: sign_change_rows (zero-crossing indicator over time, src/signchange.almd)."""
import numpy as np

NAME = "sign_change_rows"
MODULE = "signchange"
CALL = "signchange.sign_change_rows(x)"
TOL = 1e-9
SEED = 20260950


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    s = np.sign(x)
    out = np.zeros_like(x, dtype=float)
    out[1:] = (s[1:] != s[:-1]).astype(float)
    return out
