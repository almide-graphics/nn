"""Recipe: log_quad (Cauchy NLL log(1+x^2), src/logquad.almd)."""
import numpy as np

NAME = "log_quad"
MODULE = "logquad"
CALL = "logquad.log_quad(x)"
TOL = 1e-9
SEED = 20260825


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.log(1.0 + x * x)
