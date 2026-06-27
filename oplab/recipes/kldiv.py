"""Recipe: kl_div_rows (row-wise KL(p||q), src/kldiv.almd)."""
import numpy as np

NAME = "kl_div_rows"
MODULE = "kldiv"
CALL = "kldiv.kl_div_rows(p, q)"
TOL = 1e-9
SEED = 20260872


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def make_inputs(rng):
    # positive probability rows so log(p), log(q) are finite
    p = _softmax(rng.standard_normal((5, 4)))
    q = _softmax(rng.standard_normal((5, 4)))
    return {"p": ("matrix", p), "q": ("matrix", q)}


def reference(p, q):
    return (p * (np.log(p) - np.log(q))).sum(axis=1, keepdims=True)
