"""Recipe: log1p_abs (log(1+|x|), src/log1pabs.almd)."""
import numpy as np

NAME = "log1p_abs"
MODULE = "log1pabs"
CALL = "log1pabs.log1p_abs(x)"
TOL = 1e-9
SEED = 20260763


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.log(1.0 + np.abs(x))
