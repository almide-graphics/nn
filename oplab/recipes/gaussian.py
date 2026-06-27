"""Recipe: gaussian (RBF activation exp(-x^2), src/gaussian.almd)."""
import numpy as np

NAME = "gaussian"
MODULE = "gaussian"
CALL = "gaussian.gaussian(x)"
TOL = 1e-9
SEED = 20260745


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-x * x)
