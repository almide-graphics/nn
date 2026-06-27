"""Recipe: reciprocal (elementwise 1/x, src/reciprocal.almd)."""
import numpy as np

NAME = "reciprocal"
MODULE = "reciprocal"
CALL = "reciprocal.reciprocal(x)"
TOL = 1e-9
SEED = 20260759


def make_inputs(rng):
    sign = np.where(rng.standard_normal((4, 4)) > 0, 1.0, -1.0)
    mag = rng.uniform(0.5, 2.0, (4, 4))     # keep |x| away from 0
    return {"x": ("matrix", sign * mag)}


def reference(x):
    return 1.0 / x
