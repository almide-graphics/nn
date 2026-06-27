"""Recipe: sigmoid_act (logistic sigmoid, src/sigmoidact.almd)."""
import numpy as np

NAME = "sigmoid_act"
MODULE = "sigmoidact"
CALL = "sigmoidact.sigmoid_act(x)"
TOL = 1e-9
SEED = 20260757


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return 1.0 / (1.0 + np.exp(-x))
