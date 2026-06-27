"""Recipe: gaussian_times_x (x*exp(-x^2/2), src/gaussx.almd)."""
import numpy as np

NAME = "gaussian_times_x"
MODULE = "gaussx"
CALL = "gaussx.gaussian_times_x(x)"
TOL = 1e-9
SEED = 20260809


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return x * np.exp(-x * x / 2.0)
