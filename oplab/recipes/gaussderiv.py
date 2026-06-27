"""Recipe: gaussian_deriv (-x*exp(-x^2), src/gaussderiv.almd)."""
import numpy as np

NAME = "gaussian_deriv"
MODULE = "gaussderiv"
CALL = "gaussderiv.gaussian_deriv(x)"
TOL = 1e-9
SEED = 20260799


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return -x * np.exp(-x * x)
