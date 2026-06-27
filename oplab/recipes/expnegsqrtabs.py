"""Recipe: exp_neg_sqrt_abs (sub-exp bump exp(-sqrt(|x|)), src/expnegsqrtabs.almd)."""
import numpy as np

NAME = "exp_neg_sqrt_abs"
MODULE = "expnegsqrtabs"
CALL = "expnegsqrtabs.exp_neg_sqrt_abs(x)"
TOL = 1e-9
SEED = 20260841


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-np.sqrt(np.abs(x)))
