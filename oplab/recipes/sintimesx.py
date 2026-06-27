"""Recipe: sin_times_x (x*sin(x), src/sintimesx.almd)."""
import numpy as np

NAME = "sin_times_x"
MODULE = "sintimesx"
CALL = "sintimesx.sin_times_x(x)"
TOL = 1e-9
SEED = 20260815


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return x * np.sin(x)
