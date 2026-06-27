"""Recipe: expsin (exp(sin(x)), src/expsin.almd)."""
import numpy as np

NAME = "expsin"
MODULE = "expsin"
CALL = "expsin.expsin(x)"
TOL = 1e-9
SEED = 20260806


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(np.sin(x))
