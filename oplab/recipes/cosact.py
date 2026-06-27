"""Recipe: cos_act (elementwise cos(x), src/cosact.almd)."""
import numpy as np

NAME = "cos_act"
MODULE = "cosact"
CALL = "cosact.cos_act(x)"
TOL = 1e-9
SEED = 20260766


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.cos(x)
