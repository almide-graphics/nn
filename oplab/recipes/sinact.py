"""Recipe: sin_act (elementwise sin(x), src/sinact.almd)."""
import numpy as np

NAME = "sin_act"
MODULE = "sinact"
CALL = "sinact.sin_act(x)"
TOL = 1e-9
SEED = 20260765


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.sin(x)
