"""Recipe: log_softmax (row-wise, stable, src/logsoftmax.almd)."""
import numpy as np

NAME = "logsoftmax"
MODULE = "logsoftmax"
CALL = "logsoftmax.log_softmax(x)"
TOL = 1e-9
SEED = 20260704


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 5)))}


def reference(x):
    m = x.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))
    return x - lse
