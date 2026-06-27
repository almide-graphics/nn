"""Recipe: log10_act (log10(1+|x|), src/log10act.almd)."""
import numpy as np

NAME = "log10_act"
MODULE = "log10act"
CALL = "log10act.log10_act(x)"
TOL = 1e-9
SEED = 20260774


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.log10(np.abs(x) + 1.0)
