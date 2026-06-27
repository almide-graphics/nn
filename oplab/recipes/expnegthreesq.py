"""Recipe: exp_neg_three_sq (tight Gaussian exp(-3x^2), src/expnegthreesq.almd)."""
import numpy as np

NAME = "exp_neg_three_sq"
MODULE = "expnegthreesq"
CALL = "expnegthreesq.exp_neg_three_sq(x)"
TOL = 1e-9
SEED = 20260854


def make_inputs(rng):
    return {"x": ("matrix", 1.2 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-3.0 * x * x)
