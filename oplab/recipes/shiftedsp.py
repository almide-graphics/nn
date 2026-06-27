"""Recipe: shifted_softplus (softplus(x)-log2, src/shiftedsp.almd)."""
import numpy as np

NAME = "shifted_softplus"
MODULE = "shiftedsp"
CALL = "shiftedsp.shifted_softplus(x)"
TOL = 1e-9
SEED = 20260751


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    # mirror the Almide stable softplus exactly
    sp = np.maximum(x, 0.0) + np.log(1.0 + np.exp(-np.abs(x)))
    return sp - 0.6931471805599453
