"""Recipe: log2_quad (log2(1+x^2), src/log2quad.almd)."""
import numpy as np

NAME = "log2_quad"
MODULE = "log2quad"
CALL = "log2quad.log2_quad(x)"
TOL = 1e-9
SEED = 20260834


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.log2(1.0 + x * x)
