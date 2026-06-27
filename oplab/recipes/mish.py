"""Recipe: mish (self-regularized activation x·tanh(softplus(x)), src/mish.almd)."""
import numpy as np

NAME = "mish"
MODULE = "mish"
CALL = "mish.mish(x)"
TOL = 1e-9
SEED = 20260735


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    sp = np.maximum(x, 0.0) + np.log(1.0 + np.exp(-np.abs(x)))   # stable softplus
    return x * np.tanh(sp)
