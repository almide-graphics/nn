"""Recipe: log10_quad (log10(1+x^2), src/log10quad.almd)."""
import numpy as np

NAME = "log10_quad"
MODULE = "log10quad"
CALL = "log10quad.log10_quad(x)"
TOL = 1e-9
SEED = 20260832


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.log10(1.0 + x * x)
