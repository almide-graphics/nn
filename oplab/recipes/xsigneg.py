"""Recipe: x_times_sigmoid_neg (x*sigmoid(-x) = x/(1+e^x), src/xsigneg.almd)."""
import numpy as np

NAME = "x_times_sigmoid_neg"
MODULE = "xsigneg"
CALL = "xsigneg.x_times_sigmoid_neg(x)"
TOL = 1e-9
SEED = 20260811


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return x / (1.0 + np.exp(x))
