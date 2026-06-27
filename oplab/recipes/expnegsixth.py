"""Recipe: exp_neg_sixth (high-order super-Gaussian exp(-x^6), src/expnegsixth.almd)."""
import numpy as np

NAME = "exp_neg_sixth"
MODULE = "expnegsixth"
CALL = "expnegsixth.exp_neg_sixth(x)"
TOL = 1e-9
SEED = 20260870


def make_inputs(rng):
    return {"x": ("matrix", 1.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-x ** 6)
