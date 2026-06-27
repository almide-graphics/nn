"""Recipe: logsumexp_time_rows (log-softmax over the time axis, src/logsumexptime.almd)."""
import numpy as np

NAME = "logsumexp_time_rows"
MODULE = "logsumexptime"
CALL = "logsumexptime.logsumexp_time_rows(x)"
TOL = 1e-9
SEED = 20260915


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    m = x.max(axis=0, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=0, keepdims=True)))
