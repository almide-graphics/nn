"""Recipe: gompertz (double-exponential sigmoid exp(-exp(-x)), src/gompertz.almd)."""
import numpy as np

NAME = "gompertz"
MODULE = "gompertz"
CALL = "gompertz.gompertz(x)"
TOL = 1e-9
SEED = 20260791


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-np.exp(-x))
