"""Recipe: exponential_act (elementwise exp(x), src/expact.almd)."""
import numpy as np

NAME = "exponential_act"
MODULE = "expact"
CALL = "expact.exponential_act(x)"
TOL = 1e-9
SEED = 20260755


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((4, 4)))}   # moderate, no overflow


def reference(x):
    return np.exp(x)
