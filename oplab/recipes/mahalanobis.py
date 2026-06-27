"""Recipe: mahalanobis_diag_rows (diagonal Mahalanobis distance, variance-normalized Euclidean, src/mahalanobis.almd)."""
import numpy as np

NAME = "mahalanobis_diag_rows"
MODULE = "mahalanobis"
CALL = "mahalanobis.mahalanobis_diag_rows(a, b, v)"
TOL = 1e-9
SEED = 20261045


def make_inputs(rng):
    a = rng.standard_normal((5, 4))
    b = rng.standard_normal((5, 4))
    v = rng.standard_normal((5, 4)) ** 2  # per-feature variance, non-negative
    return {"a": ("matrix", a), "b": ("matrix", b), "v": ("matrix", v)}


def reference(a, b, v):
    return np.sqrt(((a - b) ** 2 / (v + 1e-12)).sum(axis=1, keepdims=True))
