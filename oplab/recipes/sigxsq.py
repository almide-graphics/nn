"""Recipe: sigmoid_times_x_sq (x^2 * sigmoid(x), src/sigxsq.almd)."""
import numpy as np

NAME = "sigmoid_times_x_sq"
MODULE = "sigxsq"
CALL = "sigxsq.sigmoid_times_x_sq(x)"
TOL = 1e-9
SEED = 20260831


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return x * x / (1.0 + np.exp(-x))
