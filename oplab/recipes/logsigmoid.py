"""Recipe: logsigmoid (log sigmoid, src/logsigmoid.almd)."""
import numpy as np

NAME = "logsigmoid"
MODULE = "logsigmoid"
CALL = "logsigmoid.logsigmoid(x)"
TOL = 1e-9
SEED = 20260754


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return -np.logaddexp(0.0, -x)   # = log sigmoid(x), stable
