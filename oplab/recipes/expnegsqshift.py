"""Recipe: exp_neg_sq_shift (RBF centered at 1, exp(-(x-1)^2), src/expnegsqshift.almd)."""
import numpy as np

NAME = "exp_neg_sq_shift"
MODULE = "expnegsqshift"
CALL = "expnegsqshift.exp_neg_sq_shift(x)"
TOL = 1e-9
SEED = 20260812


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-(x - 1.0) ** 2)
