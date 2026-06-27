"""Recipe: exp_neg_quartic (super-Gaussian exp(-x^4), src/expnegquartic.almd)."""
import numpy as np

NAME = "exp_neg_quartic"
MODULE = "expnegquartic"
CALL = "expnegquartic.exp_neg_quartic(x)"
TOL = 1e-9
SEED = 20260836


def make_inputs(rng):
    return {"x": ("matrix", 1.3 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-x ** 4)
