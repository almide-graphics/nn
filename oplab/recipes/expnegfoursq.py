"""Recipe: exp_neg_four_sq (tightest Gaussian exp(-4x^2), src/expnegfoursq.almd)."""
import numpy as np

NAME = "exp_neg_four_sq"
MODULE = "expnegfoursq"
CALL = "expnegfoursq.exp_neg_four_sq(x)"
TOL = 1e-9
SEED = 20260861


def make_inputs(rng):
    return {"x": ("matrix", 1.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-4.0 * x * x)
