"""Recipe: corr_rows (row-wise Pearson correlation, src/corrrows.almd)."""
import numpy as np

NAME = "corr_rows"
MODULE = "corrrows"
CALL = "corrrows.corr_rows(a, b)"
TOL = 1e-9
SEED = 20260887


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    da = a - a.mean(axis=1, keepdims=True)
    db = b - b.mean(axis=1, keepdims=True)
    cov = (da * db).mean(axis=1, keepdims=True)
    va = (da * da).mean(axis=1, keepdims=True)
    vb = (db * db).mean(axis=1, keepdims=True)
    return cov / (np.sqrt(va) * np.sqrt(vb))
