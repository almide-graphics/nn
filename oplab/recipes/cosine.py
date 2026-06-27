"""Recipe: cosine_sim_rows (row-wise cosine similarity, src/cosine.almd)."""
import numpy as np

NAME = "cosine_sim_rows"
MODULE = "cosine"
CALL = "cosine.cosine_sim_rows(a, b)"
TOL = 1e-9
SEED = 20260716


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    dot = (a * b).sum(axis=1, keepdims=True)
    na = np.sqrt((a * a).sum(axis=1, keepdims=True))
    nb = np.sqrt((b * b).sum(axis=1, keepdims=True))
    return dot / (na * nb)
