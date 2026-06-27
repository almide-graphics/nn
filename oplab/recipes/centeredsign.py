"""Recipe: centered_sign_rows (sign of each entry vs its own row mean, axis=1, src/centeredsign.almd)."""
import numpy as np

NAME = "centered_sign_rows"
MODULE = "centeredsign"
CALL = "centeredsign.centered_sign_rows(x)"
TOL = 1e-9
SEED = 20260955


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    mu = x.mean(axis=1, keepdims=True)
    return np.sign(x - mu)
