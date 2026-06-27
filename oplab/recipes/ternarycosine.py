"""Recipe: ternary_cosine_rows (row-wise cosine similarity of two matrices' TWN ternary codes, src/ternarycosine.almd)."""
import numpy as np

NAME = "ternary_cosine_rows"
MODULE = "ternarycosine"
CALL = "ternarycosine.ternary_cosine_rows(a, b)"
TOL = 1e-9
SEED = 20260967


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    da = 0.7 * np.abs(a).mean(axis=1, keepdims=True)
    db = 0.7 * np.abs(b).mean(axis=1, keepdims=True)
    ca = np.where(a > da, 1.0, np.where(a < -da, -1.0, 0.0))
    cb = np.where(b > db, 1.0, np.where(b < -db, -1.0, 0.0))
    dot = (ca * cb).sum(axis=1, keepdims=True)
    na = (ca * ca).sum(axis=1, keepdims=True)
    nb = (cb * cb).sum(axis=1, keepdims=True)
    return dot / (np.sqrt(na * nb) + 1e-12)
