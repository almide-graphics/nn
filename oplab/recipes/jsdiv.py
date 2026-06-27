"""Recipe: js_div_rows (row-wise Jensen-Shannon divergence, src/jsdiv.almd)."""
import numpy as np

NAME = "js_div_rows"
MODULE = "jsdiv"
CALL = "jsdiv.js_div_rows(p, q)"
TOL = 1e-9
SEED = 20260878


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def make_inputs(rng):
    p = _softmax(rng.standard_normal((5, 4)))
    q = _softmax(rng.standard_normal((5, 4)))
    return {"p": ("matrix", p), "q": ("matrix", q)}


def reference(p, q):
    m = 0.5 * (p + q)
    # mirror the op's per-element contribution then sum
    contrib = 0.5 * p * (np.log(p) - np.log(m)) + 0.5 * q * (np.log(q) - np.log(m))
    return contrib.sum(axis=1, keepdims=True)
