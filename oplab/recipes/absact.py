"""Recipe: abs_act (elementwise |x|, src/absact.almd)."""
import numpy as np

NAME = "abs_act"
MODULE = "absact"
CALL = "absact.abs_act(x)"
TOL = 1e-9
SEED = 20260753


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.abs(x)
