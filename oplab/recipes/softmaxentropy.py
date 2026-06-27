"""Recipe: softmax_entropy_rows (Shannon entropy of the softmax of each logit row, src/softmaxentropy.almd)."""
import numpy as np

NAME = "softmax_entropy_rows"
MODULE = "softmaxentropy"
CALL = "softmaxentropy.softmax_entropy_rows(x)"
TOL = 1e-9
SEED = 20260968


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return -(p * np.log(p + 1e-12)).sum(axis=1, keepdims=True)
