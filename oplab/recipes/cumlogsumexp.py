"""Recipe: cumulative_logsumexp_rows (cumulative log-sum-exp over time, src/cumlogsumexp.almd)."""
import numpy as np

NAME = "cumulative_logsumexp_rows"
MODULE = "cumlogsumexp"
CALL = "cumlogsumexp.cumulative_logsumexp_rows(x)"
TOL = 1e-9
SEED = 20261007


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.log(np.cumsum(np.exp(x), axis=0))
