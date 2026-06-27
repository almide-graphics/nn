"""Recipe: ruzicka_similarity_rows (Ruzicka / weighted Jaccard similarity sum(min)/sum(max), src/ruzicka.almd)."""
import numpy as np

NAME = "ruzicka_similarity_rows"
MODULE = "ruzicka"
CALL = "ruzicka.ruzicka_similarity_rows(a, b)"
TOL = 1e-9
SEED = 20261052


def make_inputs(rng):
    a = rng.standard_normal((5, 4)) ** 2  # non-negative
    b = rng.standard_normal((5, 4)) ** 2
    return {"a": ("matrix", a), "b": ("matrix", b)}


def reference(a, b):
    smin = np.minimum(a, b).sum(axis=1, keepdims=True)
    smax = np.maximum(a, b).sum(axis=1, keepdims=True) + 1e-12
    return smin / smax
