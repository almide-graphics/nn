"""Recipe: abs_times_sigmoid (|x|*sigmoid(x), src/abssig.almd)."""
import numpy as np

NAME = "abs_times_sigmoid"
MODULE = "abssig"
CALL = "abssig.abs_times_sigmoid(x)"
TOL = 1e-9
SEED = 20260849


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.abs(x) / (1.0 + np.exp(-x))
