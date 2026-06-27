"""Recipe: soft_cross_entropy_rows (cross-entropy between two probability distributions, src/softcrossentropy.almd)."""
import numpy as np

NAME = "soft_cross_entropy_rows"
MODULE = "softcrossentropy"
CALL = "softcrossentropy.soft_cross_entropy_rows(p, q)"
TOL = 1e-9
SEED = 20261034


def make_inputs(rng):
    p = rng.random((5, 4))
    p = p / p.sum(axis=1, keepdims=True)
    q = rng.random((5, 4))
    q = q / q.sum(axis=1, keepdims=True)
    return {"p": ("matrix", p), "q": ("matrix", q)}


def reference(p, q):
    return -(p * np.log(q + 1e-12)).sum(axis=1, keepdims=True)
