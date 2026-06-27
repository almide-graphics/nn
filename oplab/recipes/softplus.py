"""Recipe: softplus (stable log(1+exp x), src/softplus.almd)."""
import numpy as np

NAME = "softplus"
MODULE = "softplus"
CALL = "softplus.softplus(x)"
TOL = 1e-9
SEED = 20260705


def make_inputs(rng):
    return {"x": ("matrix", 2.5 * rng.standard_normal((4, 3)))}


def reference(x):
    # mirror the Almide stable form exactly
    return np.maximum(x, 0.0) + np.log(1.0 + np.exp(-np.abs(x)))
