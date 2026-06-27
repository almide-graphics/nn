"""Recipe: exp_neg_sqrt2_abs (√2-scaled sub-exp exp(-sqrt(2|x|)), src/expnegsqrt2abs.almd)."""
import numpy as np

NAME = "exp_neg_sqrt2_abs"
MODULE = "expnegsqrt2abs"
CALL = "expnegsqrt2abs.exp_neg_sqrt2_abs(x)"
TOL = 1e-9
SEED = 20260867


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-np.sqrt(2.0 * np.abs(x)))
