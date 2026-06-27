"""Recipe: abs_smooth (Charbonnier sqrt(x^2+eps), src/abssmooth.almd)."""
import numpy as np

NAME = "abs_smooth"
MODULE = "abssmooth"
CALL = "abssmooth.abs_smooth(x)"
TOL = 1e-9
SEED = 20260817


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.sqrt(x * x + 1e-4)
