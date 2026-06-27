"""Recipe: sinh_act (hyperbolic sine, src/sinhact.almd)."""
import numpy as np

NAME = "sinh_act"
MODULE = "sinhact"
CALL = "sinhact.sinh_act(x)"
TOL = 1e-9
SEED = 20260777


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((4, 4)))}


def reference(x):
    return np.sinh(x)
