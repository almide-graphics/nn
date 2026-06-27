"""Recipe: mean_sign_rows (per-row mean of signs = sign balance, src/meansign.almd)."""
import numpy as np

NAME = "mean_sign_rows"
MODULE = "meansign"
CALL = "meansign.mean_sign_rows(x)"
TOL = 1e-9
SEED = 20260948


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.sign(x).mean(axis=1, keepdims=True)
