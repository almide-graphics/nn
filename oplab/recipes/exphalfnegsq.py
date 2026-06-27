"""Recipe: exp_half_neg_sq (unit-sigma Gaussian exp(-x^2/2), src/exphalfnegsq.almd)."""
import numpy as np

NAME = "exp_half_neg_sq"
MODULE = "exphalfnegsq"
CALL = "exphalfnegsq.exp_half_neg_sq(x)"
TOL = 1e-9
SEED = 20260820


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-0.5 * x * x)
