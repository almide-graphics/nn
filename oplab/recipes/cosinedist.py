"""Recipe: cosine_dist_rows (row-wise cosine distance, src/cosinedist.almd)."""
import numpy as np

NAME = "cosine_dist_rows"
MODULE = "cosinedist"
CALL = "cosinedist.cosine_dist_rows(a, b)"
TOL = 1e-9
SEED = 20260879


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    dot = (a * b).sum(axis=1, keepdims=True)
    na = np.sqrt((a * a).sum(axis=1, keepdims=True))
    nb = np.sqrt((b * b).sum(axis=1, keepdims=True))
    return 1.0 - dot / (na * nb)
