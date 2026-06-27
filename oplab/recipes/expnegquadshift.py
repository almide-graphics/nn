"""Recipe: exp_neg_quad_shift (RBF centered at 2, exp(-(x-2)^2), src/expnegquadshift.almd)."""
import numpy as np

NAME = "exp_neg_quad_shift"
MODULE = "expnegquadshift"
CALL = "expnegquadshift.exp_neg_quad_shift(x)"
TOL = 1e-9
SEED = 20260851


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-(x - 2.0) ** 2)
