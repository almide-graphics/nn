"""Recipe: sine_sq (sin^2(x), src/sinesq.almd)."""
import numpy as np

NAME = "sine_sq"
MODULE = "sinesq"
CALL = "sinesq.sine_sq(x)"
TOL = 1e-9
SEED = 20260839


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.sin(x) ** 2
