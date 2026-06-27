"""Recipe: abs_times_tanh (|x|*tanh(x), src/abstanh.almd)."""
import numpy as np

NAME = "abs_times_tanh"
MODULE = "abstanh"
CALL = "abstanh.abs_times_tanh(x)"
TOL = 1e-9
SEED = 20260855


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.abs(x) * np.tanh(x)
