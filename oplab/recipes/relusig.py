"""Recipe: relu_times_sigmoid (max(0,x)*sigmoid(x), src/relusig.almd)."""
import numpy as np

NAME = "relu_times_sigmoid"
MODULE = "relusig"
CALL = "relusig.relu_times_sigmoid(x)"
TOL = 1e-9
SEED = 20260843


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.maximum(0.0, x) / (1.0 + np.exp(-x))
