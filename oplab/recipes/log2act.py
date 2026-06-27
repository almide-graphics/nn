"""Recipe: log2_act (log2(1+|x|), src/log2act.almd)."""
import numpy as np

NAME = "log2_act"
MODULE = "log2act"
CALL = "log2act.log2_act(x)"
TOL = 1e-9
SEED = 20260771


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.log2(np.abs(x) + 1.0)
