"""Recipe: exp_neg_half_abs (scale-2 Laplace exp(-|x|/2), src/expneghalfabs.almd)."""
import numpy as np

NAME = "exp_neg_half_abs"
MODULE = "expneghalfabs"
CALL = "expneghalfabs.exp_neg_half_abs(x)"
TOL = 1e-9
SEED = 20260845


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-0.5 * np.abs(x))
