"""Recipe: tanh_act (tanh activation, src/tanhact.almd)."""
import numpy as np

NAME = "tanh_act"
MODULE = "tanhact"
CALL = "tanhact.tanh_act(x)"
TOL = 1e-9
SEED = 20260740


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.tanh(x)
