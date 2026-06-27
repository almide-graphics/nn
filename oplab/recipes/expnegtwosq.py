"""Recipe: exp_neg_two_sq (narrow Gaussian exp(-2x^2), src/expnegtwosq.almd)."""
import numpy as np

NAME = "exp_neg_two_sq"
MODULE = "expnegtwosq"
CALL = "expnegtwosq.exp_neg_two_sq(x)"
TOL = 1e-9
SEED = 20260848


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-2.0 * x * x)
