"""Recipe: log1p_exp_neg (softplus residual log(1+exp(-|x|)), src/log1pexpneg.almd)."""
import numpy as np

NAME = "log1p_exp_neg"
MODULE = "log1pexpneg"
CALL = "log1pexpneg.log1p_exp_neg(x)"
TOL = 1e-9
SEED = 20260810


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.log(1.0 + np.exp(-np.abs(x)))
