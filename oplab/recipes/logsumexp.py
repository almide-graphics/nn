"""Recipe: logsumexp_rows (stable row-wise log-sum-exp, src/logsumexp.almd)."""
import numpy as np

NAME = "logsumexp_rows"
MODULE = "logsumexp"
CALL = "logsumexp.logsumexp_rows(x)"
TOL = 1e-9
SEED = 20260734


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((5, 4)))}


def reference(x):
    m = x.max(axis=1, keepdims=True)
    return m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))
