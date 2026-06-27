"""Recipe: sqrt_abs (sqrt(|x|), src/sqrtabs.almd)."""
import numpy as np

NAME = "sqrt_abs"
MODULE = "sqrtabs"
CALL = "sqrtabs.sqrt_abs(x)"
TOL = 1e-9
SEED = 20260761


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.sqrt(np.abs(x))
