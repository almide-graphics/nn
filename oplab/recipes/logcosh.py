"""Recipe: logcosh (log(cosh(x)) Huber-like loss, src/logcosh.almd)."""
import numpy as np

NAME = "logcosh"
MODULE = "logcosh"
CALL = "logcosh.logcosh(x)"
TOL = 1e-9
SEED = 20260783


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((4, 4)))}


def reference(x):
    return np.log(np.cosh(x))
